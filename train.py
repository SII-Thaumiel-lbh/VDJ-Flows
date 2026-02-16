#!/usr/bin/env python
# coding: utf-8
"""
Training Script with Resume Capability
Uses real datasets and Hugging Face Accelerate with checkpoint loading.
"""
import os
import sys
from collections import defaultdict
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

# 导入真实的模块和函数
from model.model import ConditionalEditFlowsTransformer
from utils.train_utils import make_ut_mask_from_z, fill_gap_tokens_with_repeats, rm_gap_tokens
from data.dataloader import SimpleTokenizer, OAS_dataloader
from utils.flow import CubicScheduler, KappaScheduler, x2prob, sample_p 


def sample_cond_pt(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, kappa: KappaScheduler):
    """Sample from conditional probability path in Z space"""
    device = p0.device
    t = t.to(device).reshape(-1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt)


def train(
    model: ConditionalEditFlowsTransformer,
    optimizer: Optimizer,
    accelerator: Accelerator,
    x_1_loader: DataLoader,
    config: dict,
    start_step: int = 0, # 新增：起始步数
):
    """
    Main training loop with real data and resume capability
    """
    # Extract config
    batch_size = config['batch_size']
    num_steps = config['num_steps']
    save_metrics = config['save_metrics']
    save_steps = config.get('save_steps', None)
    save_path_base = config.get('save_path_base', 'checkpoint.pt')
    tokenizer = config['tokenizer']
    vocab_size = tokenizer.actual_vocab_size
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    gap_token_id = tokenizer.gap_token_id
    log_dir = config.get('log_dir', 'runs')
    
    writer = None
    if accelerator.is_main_process and save_metrics:
        writer = SummaryWriter(log_dir=log_dir)
    
    scheduler = CubicScheduler(a=1.0, b=1.0)
    model.train()
    
    num_processes = accelerator.num_processes
    effective_batch_size = batch_size * num_processes
    
    # 计算剩余迭代次数
    total_iterations = num_steps // effective_batch_size
    remaining_iterations = total_iterations - (start_step // effective_batch_size)
    
    if accelerator.is_main_process:
        print(f"Distributed Training Info:")
        print(f"  Start Step: {start_step}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Remaining training iterations: {remaining_iterations}")
    
    pbar = tqdm(x_1_loader, desc="Training VDJ-Flows", unit="iter", total=total_iterations, 
                disable=not accelerator.is_main_process)
    
    # 如果是恢复训练，进度条跳到起始位置
    if start_step > 0:
        pbar.update(start_step // effective_batch_size)

    metrics = defaultdict(list)
    lr_scheduler = config.get('lr_scheduler', None)
    
    for i, (x_0, x_1, z_0, z_1, t, _, vdj_embs) in enumerate(pbar):
        # 计算全局步数
        global_step = start_step + i * effective_batch_size
        
        if global_step >= num_steps:
            break
            
        x_0, x_1, z_0, z_1, t = x_0.to(accelerator.device), x_1.to(accelerator.device), \
                               z_0.to(accelerator.device), z_1.to(accelerator.device), t.to(accelerator.device)

        # Interpolate in Z space
        num_classes_for_z = vocab_size + 3
        z_t = sample_cond_pt(x2prob(z_0, num_classes_for_z), x2prob(z_1, num_classes_for_z), t, scheduler)
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, pad_token=pad_token_id, gap_token=gap_token_id)
        # Create mask for correct edit operations
        uz_mask = make_ut_mask_from_z(z_t, z_1, vocab_size=vocab_size+2, pad_token=pad_token_id, gap_token=gap_token_id).to(accelerator.device)
        # Forward
        u_t, ins_probs, sub_probs = model.forward(
            tokens=x_t, 
            VDJ_embeddings=vdj_embs,
            time_step=t, 
            padding_mask=x_pad_mask
        )
        # Construct joint edit rate vectors
        lambda_ins, lambda_sub, lambda_del = u_t[:, :, 0], u_t[:, :, 1], u_t[:, :, 2]
        ux_cat = torch.cat([lambda_ins.unsqueeze(-1) * ins_probs, lambda_sub.unsqueeze(-1) * sub_probs, lambda_del.unsqueeze(-1)], dim=-1)
        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap_mask, z_pad_mask).to(accelerator.device)
        u_tot = u_t.sum(dim=(1, 2))
        # Compute Bregman divergence loss
        kappa_t = scheduler(t)
        kappa_derivative = scheduler.derivative(t)
        sched_coeff = (kappa_derivative / (1 - kappa_t))
        reward = (torch.clamp(uz_cat.log(), min=-20) * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        loss = (u_tot - reward).mean()
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if accelerator.is_main_process:
            u_con = (uz_cat * uz_mask).sum(dim=(1, 2)).mean().detach().cpu()
            if writer is not None:
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Metrics/u_tot', u_tot.mean().item(), global_step)
                writer.add_scalar('Metrics/u_ins', lambda_ins.sum(dim=1).mean().detach().cpu().item(), global_step)
                writer.add_scalar('Metrics/u_del', lambda_del.sum(dim=1).mean().detach().cpu().item(), global_step)
                writer.add_scalar('Metrics/u_sub', lambda_sub.sum(dim=1).mean().detach().cpu().item(), global_step)
                writer.add_scalar('Metrics/u_con', u_con.item(), global_step)
                u_con_u_tot_ratio_mean = (u_con / (u_tot.mean() + 1e-8)).item()
                writer.add_scalar('Metrics/u_con_u_tot_ratio', u_con_u_tot_ratio_mean, global_step)
                writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)

        if save_steps is not None and global_step > 0 and global_step % save_steps == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = save_path_base.replace('.pt', f'_step_{global_step}.pt')
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                    'step': global_step,
                    'vocab_size': unwrapped_model.vocab_size,
                    'actual_vocab_size': tokenizer.actual_vocab_size,
                    'hidden_dim': unwrapped_model.hidden_dim,
                    'num_layers': unwrapped_model.num_layers,
                    'num_heads': unwrapped_model.num_heads,
                    'max_seq_len': unwrapped_model.max_seq_len,
                }, save_path)
                print(f"\nSaved checkpoint: {save_path}")

    if writer is not None: writer.close()
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VDJ-Flows Model with Resume')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--vdj_emb_dict_path', type=str, required=True)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_path', type=str, default='checkpoint.pt')
    parser.add_argument('--save_steps', type=int, default=50000)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=192)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--shuffle_files', action='store_true')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--mixed_precision', type=str, default=None, choices=['no', 'fp16', 'bf16'], 
                        help='Mixed precision training: no, fp16, or bf16')
    parser.add_argument('--use_flash_attention', action='store_true', 
                        help='Use Flash Attention for faster training (requires PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # 设置混合精度
    mixed_precision = args.mixed_precision if args.mixed_precision != 'no' else None
    
    accelerator = Accelerator(split_batches=False, mixed_precision=mixed_precision)
    
    if accelerator.is_main_process:
        if mixed_precision:
            print(f"Mixed precision training enabled: {mixed_precision.upper()}")
        else:
            print("Mixed precision training disabled (using FP32)")
    
    # 1. 加载 Tokenizer 和模型基础配置
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)
    
    # 加载 evo2 的 VDJ embeddings 字典
    vdj_embeddings_dict = torch.load(args.vdj_emb_dict_path, map_location='cpu')
    
    # 获取 evo2 的 VDJ embedding 维度（从第一个 embedding 获取）
    first_emb_key = next(iter(vdj_embeddings_dict.keys()))
    first_emb = vdj_embeddings_dict[first_emb_key]
    # 如果是多维tensor，取最后一维；如果是一维，取长度
    if first_emb.dim() > 1:
        vdj_embedding_dim = first_emb.shape[-1]
    else:
        vdj_embedding_dim = first_emb.shape[0]
    
    model = ConditionalEditFlowsTransformer(
        vocab_size=tokenizer.actual_vocab_size + 2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        VDJ_embedding_dim=vdj_embedding_dim,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    if accelerator.is_main_process:
        if args.use_flash_attention:
            from model.model import HAS_FLASH_ATTN
            if HAS_FLASH_ATTN:
                print("Flash Attention enabled - using optimized attention computation")
            else:
                print("Flash Attention requested but not available (requires PyTorch 2.0+)")
        else:
            print("Using standard TransformerEncoderLayer")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # resume checkpoint
    start_step = 0
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        
        if accelerator.is_main_process:
            print(f"Checkpoint loaded. Resuming from step {start_step}")

    # prepare data loader
    dataloader_wrapper = OAS_dataloader(
        base_dir=args.data_dir,
        VDJ_cols=["v_germline", "d_germline", "j_germline"],
        VDJ_germline_ids_col=["v_call", "d_call", "j_call"],
        seq_col="sequence_alignment_aa_filled",
        vdj_embeddings_dict=vdj_embeddings_dict,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        shuffle_files=args.shuffle_files,
        random_amino_acid_num=10
    )
    
    x_1_loader = dataloader_wrapper.get_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )

    # prepare learning rate scheduler
    num_iterations = args.num_steps // (args.batch_size)
    warmup_iterations = int(num_iterations * 0.05)
    
    def lr_lambda(iteration):
        if iteration < warmup_iterations:
            return iteration / warmup_iterations
        return 1.0 - 0.5 * ((iteration - warmup_iterations) / (num_iterations - warmup_iterations))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # resume learning rate scheduler
    if args.resume_from_checkpoint and 'lr_scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    # prepare everything
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    
    config = {
        'batch_size': args.batch_size,
        'tokenizer': tokenizer,
        'num_steps': args.num_steps,
        'save_metrics': True,
        'save_steps': args.save_steps,
        'save_path_base': args.save_path,
        'lr_scheduler': lr_scheduler,
        'log_dir': args.log_dir,
    }

    # start training
    train(model, optimizer, accelerator, x_1_loader, config, start_step=start_step)