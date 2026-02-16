#!/usr/bin/env python
# coding: utf-8
"""
VDJ-Conditioned Inference Script
Generate antibody sequences from VDJ germline sequences
"""

import sys
import os
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt

from model.model import ConditionalEditFlowsTransformer
from data.dataloader import SimpleTokenizer
from utils.flow import CubicScheduler
from utils.constant import germline_sequences_V, germline_sequences_J


def load_model(checkpoint_path: str, device: torch.device, vdj_embeddings_dict: dict = None) -> ConditionalEditFlowsTransformer:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get VDJ embedding dimension from checkpoint or embeddings dict
    if vdj_embeddings_dict is not None:
        first_emb_key = next(iter(vdj_embeddings_dict.keys()))
        first_emb = vdj_embeddings_dict[first_emb_key]
        if first_emb.dim() > 1:
            vdj_embedding_dim = first_emb.shape[-1]
        else:
            vdj_embedding_dim = first_emb.shape[0]
    else:
        # Try to get from checkpoint, or use default
        vdj_embedding_dim = checkpoint.get('VDJ_embedding_dim', 4096)  # Default evo2 embedding dim
    
    # Create model with saved configuration
    model = ConditionalEditFlowsTransformer(
        vocab_size=checkpoint['vocab_size'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        VDJ_embedding_dim=vdj_embedding_dim,
        num_heads=checkpoint['num_heads'],
        max_seq_len=checkpoint['max_seq_len'],
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Vocab size: {checkpoint['vocab_size']}")
    print(f"  Hidden dim: {checkpoint['hidden_dim']}")
    print(f"  Num layers: {checkpoint['num_layers']}")
    print(f"  Num heads: {checkpoint['num_heads']}")
    print(f"  Max seq len: {checkpoint['max_seq_len']}")
    print(f"  VDJ embedding dim: {vdj_embedding_dim}")
    
    return model

def apply_ins_del_operations(
    x_t: torch.Tensor,
    ins_mask: torch.Tensor,
    del_mask: torch.Tensor,
    ins_tokens: torch.Tensor,
    pad_token: int,
    max_seq_len: int = 512,
) -> torch.Tensor:
    """Apply insertion and deletion operations to sequences"""
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # Handle simultaneous ins+del as substitutions
    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]

    # Update ins/del masks after handling replacements
    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask

    # Compute new lengths
    xt_pad_mask = (x_t == pad_token)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = min(new_lengths.max().item(), max_seq_len)

    # Initialize output
    x_t_new = torch.full((batch_size, max_new_len), pad_token, dtype=torch.long, device=device)

    for b in range(batch_size):
        cur_seq = x_t_modified[b]
        cur_ins = eff_ins_mask[b]
        cur_del = eff_del_mask[b]
        cur_ins_tokens = ins_tokens[b]

        # Build new sequence
        new_seq = []
        for pos in range(seq_len):
            if cur_seq[pos] == pad_token:
                break
            # Insert token if needed
            if cur_ins[pos]:
                new_seq.append(cur_ins_tokens[pos].item())
            # Keep token if not deleted
            if not cur_del[pos]:
                new_seq.append(cur_seq[pos].item())

        new_seq = new_seq[:max_new_len]
        if new_seq:
            x_t_new[b, :len(new_seq)] = torch.tensor(new_seq, dtype=torch.long, device=device)

    return x_t_new

def get_adaptive_h(default_h: float, t: torch.Tensor, scheduler: CubicScheduler) -> torch.Tensor:
    """Get adaptive step size based on schedule"""
    # Use formula: h_adapt = min(h, (1 - kappa(t)) / kappa'(t))
    # This ensures step size never exceeds default_h
    kappa_t = scheduler(t)
    kappa_deriv = scheduler.derivative(t)
    
    # Compute coefficient: (1 - kappa(t)) / kappa'(t)
    coeff = (1 - kappa_t) / (kappa_deriv + 1e-8)
    
    # Adaptive step is minimum of default_h and coefficient
    h_tensor = default_h * torch.ones_like(t, device=t.device)
    h_adapt = torch.minimum(h_tensor, coeff)
    
    return h_adapt

@torch.no_grad()
def sample_from_vdj(
    model: ConditionalEditFlowsTransformer,
    v_calls: List[str],
    d_calls: List[str],
    j_calls: List[str],
    vdj_embeddings_dict: dict,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    n_steps: int = 1000,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    output_file: Optional[str] = None,
    save_at_times: Optional[List[float]] = None,
    track_edit_rates: bool = False,
    random_aas_num: int = 15,
) -> Tuple[List[str], List[List[torch.Tensor]], dict]:
    """
    Generate antibody sequences conditioned on VDJ germline sequences
    
    Args:
        model: Trained model
        v_calls: List of V gene names (e.g., "IGHV1-18*01")
        d_calls: List of D gene names
        j_calls: List of J gene names
        vdj_embeddings_dict: Dictionary mapping gene names to embeddings
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run on
        n_steps: Number of Euler steps
        temperature: Sampling temperature
        seed: Random seed for reproducibility
        output_file: Path to save output file
        save_at_times: List of time points (e.g., [0.5]) to save intermediate sequences
        track_edit_rates: Whether to track edit rates during inference
        random_aas_num: Number of random amino acids to insert between v_end and j_start
        
    Returns:
        generated_sequences: List of generated antibody sequences
        trajectories: List of trajectories for each sample (optional)
        intermediate_sequences: Dict mapping time points to lists of sequences
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Setup
    scheduler = CubicScheduler(a=1.0, b=1.0)
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    gap_token_id = tokenizer.gap_token_id
    vocab_size = tokenizer.actual_vocab_size
    
    batch_size = len(v_calls)
    
    # Get V and J sequences from constant.py and build v_end_j_start (V last 6 + J first 6)
    v_end_j_start = []
    v_sequences = []
    j_sequences = []
    for v_call, j_call in zip(v_calls, j_calls):
        # Get V sequence from constant.py
        if v_call not in germline_sequences_V:
            raise ValueError(f"V gene {v_call} not found in germline_sequences_V")
        v_seq = germline_sequences_V[v_call]["aa_seq"]
        v_sequences.append(v_seq)
        
        # Get J sequence from constant.py
        if j_call not in germline_sequences_J:
            raise ValueError(f"J gene {j_call} not found in germline_sequences_J")
        j_seq = germline_sequences_J[j_call]["aa_seq"]
        j_sequences.append(j_seq)

        # 在 vend 和 j start中间添加 random_aas_num 个随机的氨基酸字母
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        rng = np.random.RandomState(seed if seed is not None else None)
        random_aas = ''.join(rng.choice(amino_acids, size=random_aas_num))
        v_end_j_start.append(v_seq[-6:] + random_aas + j_seq[:6])
    
    # Build VDJ embeddings from gene calls
    vdj_embeddings = []
    for v_call, d_call, j_call in zip(v_calls, d_calls, j_calls):
        v_emb = vdj_embeddings_dict[f"V_{v_call}"].mean(dim=0)
        d_emb = vdj_embeddings_dict[f"D_{d_call}"].mean(dim=0)
        j_emb = vdj_embeddings_dict[f"J_{j_call}"].mean(dim=0)
        vdj_embeddings.append(torch.stack([v_emb, d_emb, j_emb], dim=0))
    vdj_embeddings = torch.stack(vdj_embeddings, dim=0).to(device)  # [B, 3, VDJ_embedding_dim]
    
    # Tokenize v_end_j_start sequences (x_0)
    x_0_list = [tokenizer.encode(seq, convert_to_tensor=True) for seq in v_end_j_start]
    
    # Pad x_0 to same length and add BOS
    x0_max_len = max(len(x) for x in x_0_list)
    x_0 = torch.stack([
        F.pad(x, (0, x0_max_len - len(x)), value=pad_token_id) 
        for x in x_0_list
    ]).long().to(device)
    x_0 = F.pad(x_0, (1, 0), value=bos_token_id)  # Add BOS
    
    # Initialize x_t = x_0 (start from v_end_j_start at t=0)
    # This matches training where we interpolate from x_0 (v_end_j_start) to x_1 (antibody)
    x_t = x_0.clone()
    
    print(f"Generating {batch_size} sequences with {n_steps} steps...")
    print(f"Starting from v_end_j_start sequences (x_0):")
    print(f"  x_t shape: {x_t.shape}")
    print(f"  VDJ embeddings shape: {vdj_embeddings.shape}")
    for i in range(min(3, batch_size)):
        seq_str = tokenizer.decode([t for t in x_t[i].cpu().tolist() if t not in [pad_token_id, bos_token_id]])
        print(f"  Sample {i+1}: {seq_str[:80]}{'...' if len(seq_str) > 80 else ''}")
        print(f"    V: {v_calls[i]}, D: {d_calls[i]}, J: {j_calls[i]}")
    
    # Sampling loop: evolve from x_0 (t=0) to x_1 (t=1)
    t = torch.zeros(batch_size, 1, device=device)
    default_h = 1.0 / n_steps
    
    trajectories = [[] for _ in range(batch_size)]
    step_count = 0
    
    # Track intermediate sequences at specified time points
    intermediate_sequences = {}
    if save_at_times is not None:
        save_at_times = sorted(save_at_times)
        for t_save in save_at_times:
            intermediate_sequences[t_save] = []
        # Track which time points we've already saved (per batch)
        saved_flags = {t_save: [False] * batch_size for t_save in save_at_times}
    
    # Track edit rates for visualization (similar to training)
    inference_edit_rate_data = {}
    sample_rate_every_n_steps = max(1, n_steps // 50)  # 采样约50个点
    
    with tqdm(total=n_steps, desc="Sampling") as pbar:
        while t.max() <= 1.0 - default_h:
            # Save current state
            for i in range(batch_size):
                trajectories[i].append(x_t[i].cpu().clone())
            
            # Get padding mask
            x_pad_mask = (x_t == pad_token_id)
            
            # Forward pass with VDJ embeddings as condition
            u_t, ins_probs, sub_probs = model.forward(
                tokens=x_t,
                VDJ_embeddings=vdj_embeddings,
                time_step=t,
                padding_mask=x_pad_mask,
            )
            
            # Extract rates
            lambda_ins = u_t[:, :, 0]
            lambda_sub = u_t[:, :, 1]
            lambda_del = u_t[:, :, 2]
            
            # Track edit rates for visualization (similar to training)
            if track_edit_rates and step_count % sample_rate_every_n_steps == 0:
                # 计算每个位置的总编辑率（lambda_ins + lambda_sub + lambda_del）
                # 这是模型在所有位置预测的编辑率，与训练时统计的方式一致
                position_rates = (lambda_ins + lambda_sub + lambda_del)  # (batch, seq_len)
                valid_lengths = (~x_pad_mask).sum(dim=1).cpu().numpy()  # (batch,)
                t_values_batch = t.cpu().numpy().flatten()  # (batch,)
                position_rates_batch = position_rates.cpu().numpy()  # (batch, seq_len)
                
                # 调试：检查编辑率是否有变化
                if step_count == 0 or step_count % (sample_rate_every_n_steps * 10) == 0:
                    for b in range(min(2, batch_size)):
                        valid_len = valid_lengths[b]
                        if valid_len > 0:
                            rates_sample = position_rates_batch[b, :valid_len]
                            print(f"  [Step {step_count}, Sample {b}] t={t_values_batch[b]:.4f}, "
                                  f"rates: min={rates_sample.min():.6f}, max={rates_sample.max():.6f}, "
                                  f"mean={rates_sample.mean():.6f}, std={rates_sample.std():.6f}, "
                                  f"first_5={rates_sample[:5]}")
                
                inference_edit_rate_data[step_count] = {
                    't_values': t_values_batch,
                    'position_rates': position_rates_batch,
                    'valid_lengths': valid_lengths
                }
            
            # Apply temperature
            if temperature != 1.0:
                ins_probs = F.softmax(F.log_softmax(ins_probs, dim=-1) / temperature, dim=-1)
                sub_probs = F.softmax(F.log_softmax(sub_probs, dim=-1) / temperature, dim=-1)
            
            # Get adaptive step size
            adapt_h = get_adaptive_h(default_h, t, scheduler)
            
            # Sample operations
            ins_mask = torch.rand_like(lambda_ins) < (1 - torch.exp(-adapt_h * lambda_ins))
            del_sub_mask = torch.rand_like(lambda_sub) < (1 - torch.exp(-adapt_h * (lambda_sub + lambda_del)))
            
            # Determine deletion vs substitution
            prob_del = torch.where(
                del_sub_mask, 
                lambda_del / (lambda_sub + lambda_del + 1e-8), 
                torch.zeros_like(lambda_del)
            )
            del_mask = torch.bernoulli(prob_del).bool()
            sub_mask = del_sub_mask & ~del_mask
            
            # Sample tokens
            ins_tokens = torch.full(ins_probs.shape[:2], pad_token_id, dtype=torch.long, device=device)
            sub_tokens = torch.full(sub_probs.shape[:2], pad_token_id, dtype=torch.long, device=device)
            
            non_pad_mask = ~x_pad_mask
            if non_pad_mask.any():
                ins_sampled = torch.multinomial(ins_probs[non_pad_mask], num_samples=1).squeeze(-1)
                sub_sampled = torch.multinomial(sub_probs[non_pad_mask], num_samples=1).squeeze(-1)
                ins_tokens[non_pad_mask] = ins_sampled
                sub_tokens[non_pad_mask] = sub_sampled
            
            # Apply substitutions
            x_t[sub_mask] = sub_tokens[sub_mask]
            
            # Apply insertions and deletions
            x_t = apply_ins_del_operations(
                x_t, ins_mask, del_mask, ins_tokens,
                pad_token=pad_token_id,
                max_seq_len=model.max_seq_len
            )
            
            # Check if we've crossed any save time points (before updating t)
            if save_at_times is not None:
                t_prev = t.clone()
            
            # Update time
            t = t + adapt_h
            step_count += 1
            pbar.update(1)
            
            # Check if we've crossed any save time points and save intermediate sequences
            if save_at_times is not None:
                for t_save in save_at_times:
                    for i in range(batch_size):
                        # Check if we crossed this time point
                        if not saved_flags[t_save][i] and t_prev[i, 0].item() < t_save <= t[i, 0].item():
                            # Decode intermediate middle region
                            tokens = x_t[i].cpu().numpy()
                            tokens = tokens[tokens != bos_token_id]
                            tokens = tokens[tokens != pad_token_id]
                            middle_region = tokenizer.decode(tokens.tolist())
                            # Build full antibody sequence: V[:-6] + middle_region + J[6:]
                            v_seq = v_sequences[i]
                            j_seq = j_sequences[i]
                            full_sequence = v_seq[:-6] + middle_region + j_seq[6:]
                            intermediate_sequences[t_save].append(full_sequence)
                            saved_flags[t_save][i] = True
                            print(f"\n  [t={t_save:.2f}] Sample {i+1}: {full_sequence[:100]}{'...' if len(full_sequence) > 100 else ''}")
            
            # Debug: print every 100 steps
            if step_count % 100 == 0:
                non_pad_lens = [(x_t[i] != pad_token_id).sum().item() for i in range(min(3, batch_size))]
                print(f"\n  Step {step_count}, t_max={t.max().item():.4f}, seq_lens={non_pad_lens}")
    
    print(f"\nSampling completed in {step_count} steps (expected {n_steps})")
    print(f"Final x_t shape: {x_t.shape}")
    print(f"Final sequence lengths: {[(x_t[i] != pad_token_id).sum().item() for i in range(batch_size)]}")
    
    # Ensure all samples have intermediate sequences at each time point
    # For samples that didn't cross the time point, save the sequence at the closest time
    if save_at_times is not None:
        for t_save in save_at_times:
            # Ensure we have an entry for each sample
            while len(intermediate_sequences[t_save]) < batch_size:
                # Find samples that haven't been saved yet
                for i in range(batch_size):
                    if len(intermediate_sequences[t_save]) <= i:
                        # Save current sequence as intermediate (closest we have)
                        tokens = x_t[i].cpu().numpy()
                        tokens = tokens[tokens != bos_token_id]
                        tokens = tokens[tokens != pad_token_id]
                        middle_region = tokenizer.decode(tokens.tolist())
                        # Build full antibody sequence: V[:-6] + middle_region + J[6:]
                        v_seq = v_sequences[i]
                        j_seq = j_sequences[i]
                        full_sequence = v_seq[:-6] + middle_region + j_seq[6:]
                        intermediate_sequences[t_save].append(full_sequence)
                        if not saved_flags[t_save][i]:
                            print(f"  Note: Sample {i+1} at t={t_save:.2f} saved from final state (t={t[i, 0].item():.4f})")
    
    # Decode final sequences (generated middle region)
    generated_middle_regions = []
    for i in range(batch_size):
        tokens = x_t[i].cpu().numpy()
        # Remove BOS and PAD tokens
        tokens = tokens[tokens != bos_token_id]
        tokens = tokens[tokens != pad_token_id]
        print(f"Sample {i+1}: {len(tokens)} tokens after filtering")
        middle_region = tokenizer.decode(tokens.tolist())
        generated_middle_regions.append(middle_region)
    
    # Build full antibody sequences: V[:-6] + generated_middle_region + J[6:]
    generated_sequences = []
    for i in range(batch_size):
        v_seq = v_sequences[i]
        j_seq = j_sequences[i]
        middle_region = generated_middle_regions[i]
        # Construct full antibody: V[:-6] + generated middle region + J[6:]
        full_antibody = v_seq[:-6] + middle_region + j_seq[6:]
        generated_sequences.append(full_antibody)

    # Save results to CSV file
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Save CSV file
        csv_file = output_file if output_file.endswith('.csv') else output_file.replace('.tsv', '.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            # generated_sequence is the full antibody: V[:-6] + generated_middle_region + J[6:]
            header_cols = ["v_call", "d_call", "j_call", "v_germline", "j_germline", "v_end_j_start", "generated_sequence"]
            if save_at_times is not None:
                for t_save in sorted(save_at_times):
                    header_cols.append(f"t_{t_save:.2f}_sequence")
            writer.writerow(header_cols)
            
            # Write data rows
            for i, (v_call, d_call, j_call, v_seq, j_seq, v_end_j, gen_seq) in enumerate(
                zip(v_calls, d_calls, j_calls, v_sequences, j_sequences, v_end_j_start, generated_sequences)
            ):
                row = [v_call, d_call, j_call, v_seq, j_seq, v_end_j, gen_seq]
                if save_at_times is not None:
                    for t_save in sorted(save_at_times):
                        if i < len(intermediate_sequences.get(t_save, [])):
                            row.append(intermediate_sequences[t_save][i])
                        else:
                            row.append("")
                writer.writerow(row)
        print(f"Results saved to {csv_file}")
        
        # Save FASTA file (only generated sequences)
        fasta_file = csv_file.replace('.csv', '.fasta')
        with open(fasta_file, 'w', encoding='utf-8') as f:
            for i, (v_call, d_call, j_call, gen_seq) in enumerate(
                zip(v_calls, d_calls, j_calls, generated_sequences)
            ):
                # FASTA header: sample index and VDJ information
                header = f">sample_{i+1}_V_{v_call}_D_{d_call}_J_{j_call}"
                f.write(f"{header}\n{gen_seq}\n")
        print(f"FASTA file saved to {fasta_file}")
    
    # Print intermediate sequences summary
    if save_at_times is not None and intermediate_sequences:
        print("\n" + "="*80)
        print("Intermediate Sequences Summary:")
        print("="*80)
        for t_save in sorted(intermediate_sequences.keys()):
            print(f"\nAt t = {t_save:.2f}:")
            for i, seq in enumerate(intermediate_sequences[t_save]):
                print(f"  Sample {i+1}: {seq[:100]}{'...' if len(seq) > 100 else ''}")
    
    # Plot edit rate distribution if tracking is enabled
    if track_edit_rates and len(inference_edit_rate_data) > 0:
        # TODO: Implement plot_inference_edit_rates_by_t function for visualization
        # plot_inference_edit_rates_by_t(inference_edit_rate_data, 'inference_edit_rates_by_t.png')
        print(f"\nTracked edit rates at {len(inference_edit_rate_data)} steps (plotting not implemented)")

    return generated_sequences, trajectories, intermediate_sequences


def batch_inference_from_file(
    model: ConditionalEditFlowsTransformer,
    input_file: str,
    output_file: str,
    vdj_embeddings_dict: dict,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    batch_size: int = 8,
    n_steps: int = 1000,
    temperature: float = 1.0,
    save_at_times: Optional[List[float]] = None,
    random_aas_num: int = 15,
):
    """
    Batch inference from input file
    
    Input file format (TSV or CSV):
    v_call\td_call\tj_call
    (v_call, d_call, j_call are gene names like "IGHV1-18*01")
    ...
    
    Output files:
    1. CSV file: v_call,d_call,j_call,v_germline,j_germline,v_end_j_start,generated_sequence,...
       where generated_sequence is the full antibody: V[:-6] + generated_middle_region + J[6:]
    2. FASTA file: Contains only the generated sequences in FASTA format
    """
    print(f"Reading input from {input_file}")
    
    # Read input VDJ gene calls
    vdj_inputs = []
    with open(input_file, 'r') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                v_call, d_call, j_call = parts[0], parts[1], parts[2]
                vdj_inputs.append((v_call, d_call, j_call))
            else:
                print(f"Warning: Skipping invalid line: {line.strip()}")
    
    print(f"Loaded {len(vdj_inputs)} VDJ gene calls")
    
    # Batch inference
    all_results = []
    all_intermediate_sequences = {t: [] for t in (save_at_times or [])}
    
    for i in range(0, len(vdj_inputs), batch_size):
        batch_vdj = vdj_inputs[i:i+batch_size]
        batch_v_calls = [vdj[0] for vdj in batch_vdj]
        batch_d_calls = [vdj[1] for vdj in batch_vdj]
        batch_j_calls = [vdj[2] for vdj in batch_vdj]
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(vdj_inputs)-1)//batch_size + 1}")
        generated, _, intermediate = sample_from_vdj(
            model, batch_v_calls, batch_d_calls, batch_j_calls,
            vdj_embeddings_dict, tokenizer, device,
            n_steps=n_steps, temperature=temperature,
            save_at_times=save_at_times,
            track_edit_rates=False,  # 批量推理时不追踪（避免内存问题）
            random_aas_num=random_aas_num
        )
        
        # Get sequences for output (from constant.py)
        batch_v_seqs = []
        batch_j_seqs = []
        batch_v_end_j = []
        for v_call, j_call in zip(batch_v_calls, batch_j_calls):
            v_seq = germline_sequences_V[v_call]["aa_seq"]
            j_seq = germline_sequences_J[j_call]["aa_seq"]
            batch_v_seqs.append(v_seq)
            batch_j_seqs.append(j_seq)
            batch_v_end_j.append(v_seq[-6:] + j_seq[:6])
        
        # Collect results and intermediate sequences
        for idx, ((v_call, d_call, j_call), gen_seq) in enumerate(zip(batch_vdj, generated)):
            all_results.append((v_call, d_call, j_call, batch_v_seqs[idx], batch_j_seqs[idx], batch_v_end_j[idx], gen_seq))
            # Collect intermediate sequences for this sample
            if save_at_times is not None and intermediate:
                for t_save in save_at_times:
                    seq = intermediate.get(t_save, [None] * len(batch_v_calls))[idx] if idx < len(intermediate.get(t_save, [])) else None
                    all_intermediate_sequences[t_save].append(seq if seq is not None else "")
    
    # Save results
    print(f"\nSaving results to {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Save CSV file
    csv_file = output_file if output_file.endswith('.csv') else output_file.replace('.tsv', '.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        header_cols = ["v_call", "d_call", "j_call", "v_germline", "j_germline", "v_end_j_start", "generated_sequence"]
        if save_at_times is not None:
            for t_save in sorted(save_at_times):
                header_cols.append(f"t_{t_save:.2f}_sequence")
        writer.writerow(header_cols)
        
        # Write data rows
        for idx, (v_call, d_call, j_call, v_seq, j_seq, v_end_j, gen_seq) in enumerate(all_results):
            row = [v_call, d_call, j_call, v_seq, j_seq, v_end_j, gen_seq]
            if save_at_times is not None:
                for t_save in sorted(save_at_times):
                    if idx < len(all_intermediate_sequences[t_save]):
                        row.append(all_intermediate_sequences[t_save][idx])
                    else:
                        row.append("")
            writer.writerow(row)
    
    print(f"CSV file saved to {csv_file}")
    
    # Save FASTA file (only generated sequences)
    fasta_file = csv_file.replace('.csv', '.fasta')
    with open(fasta_file, 'w', encoding='utf-8') as f:
        for idx, (v_call, d_call, j_call, v_seq, j_seq, v_end_j, gen_seq) in enumerate(all_results):
            # FASTA header: sample index and VDJ information
            header = f">sample_{idx+1}_V_{v_call}_D_{d_call}_J_{j_call}"
            f.write(f"{header}\n{gen_seq}\n")
    
    print(f"FASTA file saved to {fasta_file}")
    print(f"Done! Generated {len(all_results)} sequences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VDJ-Conditioned Antibody Generation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary json file')
    parser.add_argument('--vdj_emb_dict_path', type=str, required=True, help='Path to VDJ embeddings dictionary')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                        help='Inference mode: single or batch')
    parser.add_argument('--output_file', type=str, default=None, help='Output CSV file for results (will also generate a FASTA file)')
    
    # Single mode arguments
    parser.add_argument('--v_call', type=str, required=True, help='V gene name (e.g., IGHV1-18*01)')
    parser.add_argument('--d_call', type=str, required=True, help='D gene name')
    parser.add_argument('--j_call', type=str, required=True, help='J gene name')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples to generate')
    
    # Batch mode arguments
    parser.add_argument('--input_file', type=str, help='Input TSV or CSV file with VDJ sequences')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    
    # Sampling arguments
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of Euler steps')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save_at_times', type=str, default=None,
                        help='Comma-separated list of time points to save intermediate sequences (e.g., "0.25,0.5,0.75")')
    parser.add_argument('--track_edit_rates', action='store_true',
                        help='Track and visualize model predicted edit rates during inference')
    parser.add_argument('--random_aas_num', type=int, default=15,
                        help='Number of random amino acids to insert between v_end and j_start')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load VDJ embeddings dictionary
    print(f"Loading VDJ embeddings from {args.vdj_emb_dict_path}")
    vdj_embeddings_dict = torch.load(args.vdj_emb_dict_path, map_location='cpu')
    print(f"Loaded {len(vdj_embeddings_dict)} VDJ embeddings")
    
    # Load model
    model = load_model(args.checkpoint, device, vdj_embeddings_dict=vdj_embeddings_dict)
    
    # Load tokenizer
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)
    
    # Parse save_at_times if provided
    save_at_times = None
    if args.save_at_times:
        save_at_times = [float(t.strip()) for t in args.save_at_times.split(',')]
        print(f"Will save intermediate sequences at times: {save_at_times}")
    
    if args.mode == 'single':
        # Single inference mode
        v_calls = [args.v_call] * args.n_samples
        d_calls = [args.d_call] * args.n_samples
        j_calls = [args.j_call] * args.n_samples
        
        # Get sequences from constant.py for display
        if args.v_call not in germline_sequences_V:
            raise ValueError(f"V gene {args.v_call} not found in germline_sequences_V")
        if args.j_call not in germline_sequences_J:
            raise ValueError(f"J gene {args.j_call} not found in germline_sequences_J")
        
        v_seq = germline_sequences_V[args.v_call]["aa_seq"]
        j_seq = germline_sequences_J[args.j_call]["aa_seq"]
        v_end_j = v_seq[-6:] + j_seq[:6]
        
        print(f"\nGenerating {args.n_samples} samples from VDJ:")
        print(f"  V: {args.v_call} -> {v_seq}")
        print(f"  D: {args.d_call}")
        print(f"  J: {args.j_call} -> {j_seq}")
        print(f"  v_end_j_start: {v_end_j}\n")
        
        generated, trajectories, intermediate = sample_from_vdj(
            model, v_calls, d_calls, j_calls,
            vdj_embeddings_dict, tokenizer, device,
            n_steps=args.n_steps,
            temperature=args.temperature,
            seed=args.seed,
            output_file=args.output_file,
            save_at_times=save_at_times,
            track_edit_rates=args.track_edit_rates,
            random_aas_num=args.random_aas_num
        )
        
        print("\nGenerated sequences:")
        for i, seq in enumerate(generated):
            print(f"Sample {i+1}: {seq}")
    
    elif args.mode == 'batch':
        # Batch inference mode
        assert args.input_file and args.output_file, \
            "Must provide input_file and output_file for batch mode"
        
        batch_inference_from_file(
            model, args.input_file, args.output_file,
            vdj_embeddings_dict, tokenizer, device,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            temperature=args.temperature,
            save_at_times=save_at_times,
            random_aas_num=args.random_aas_num
        )
    
    print("\nInference complete!")

