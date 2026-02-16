#!/usr/bin/env python
# coding: utf-8
"""
Full Inference Script: VDJ EBM + CDR3 Generation
First samples VDJ combinations using VDJ EBM, then generates antibody sequences
"""

import sys
import os
import json
import csv
import torch
import numpy as np
import argparse
from typing import List, Tuple, Optional
from tqdm import tqdm

from model.VDJ_EBM import VDJEnergyModel, VDJManager
from model.model import ConditionalEditFlowsTransformer
from data.dataloader import SimpleTokenizer
from utils.constant import germline_sequences_V, germline_sequences_J
from inference_CDR3 import sample_from_vdj, load_model as load_cdr3_model


def load_ebm_model(checkpoint_path: str, vdj_embeddings_dict: dict, device: torch.device) -> VDJManager:
    """
    Load VDJ EBM model and create VDJManager
    
    Args:
        checkpoint_path: Path to VDJ EBM checkpoint
        vdj_embeddings_dict: Dictionary mapping gene names to embeddings
        device: Device to run on
        
    Returns:
        VDJManager instance
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get embedding dimension
    first_emb_key = next(iter(vdj_embeddings_dict.keys()))
    first_emb = vdj_embeddings_dict[first_emb_key]
    if first_emb.dim() > 1:
        evo2_dim = first_emb.shape[-1]
    else:
        evo2_dim = first_emb.shape[0]
    
    # Get hidden_dim from checkpoint or use default
    hidden_dim = checkpoint.get('hidden_dim', 256)
    
    # Create model
    model = VDJEnergyModel(evo2_dim=evo2_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create VDJManager
    manager = VDJManager(model, vdj_embeddings_dict, device=device)
    
    print(f"VDJ EBM Model loaded from {checkpoint_path}")
    print(f"  EVO2 embedding dimension: {evo2_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of V genes: {len(manager.v_keys)}")
    print(f"  Number of D genes: {len(manager.d_keys)}")
    print(f"  Number of J genes: {len(manager.j_keys)}")
    print(f"  Total combinations: {len(manager.all_combinations)}")
    
    return manager


@torch.no_grad()
def full_inference(
    ebm_manager: VDJManager,
    cdr3_model: ConditionalEditFlowsTransformer,
    vdj_embeddings_dict: dict,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    num_samples: int = 10,
    ebm_temperature: float = 1.0,
    cdr3_n_steps: int = 1000,
    cdr3_temperature: float = 1.0,
    seed: Optional[int] = None,
    output_file: Optional[str] = None,
    save_at_times: Optional[List[float]] = None,
    random_aas_num: int = 15,
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """
    Full inference pipeline: Sample VDJ combinations then generate antibody sequences
    
    Args:
        ebm_manager: VDJManager instance for sampling VDJ combinations
        cdr3_model: ConditionalEditFlowsTransformer for generating CDR3 sequences
        vdj_embeddings_dict: Dictionary mapping gene names to embeddings
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run on
        num_samples: Number of VDJ combinations to sample
        ebm_temperature: Temperature for VDJ EBM sampling
        cdr3_n_steps: Number of Euler steps for CDR3 generation
        cdr3_temperature: Temperature for CDR3 generation
        seed: Random seed for reproducibility
        output_file: Path to save output file
        save_at_times: List of time points to save intermediate sequences
        random_aas_num: Number of random amino acids to insert between v_end and j_start
        
    Returns:
        sampled_vdj: List of (v_call, d_call, j_call) tuples
        generated_sequences: List of generated antibody sequences
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    print(f"\n{'='*80}")
    print("Step 1: Sampling VDJ combinations using VDJ EBM")
    print(f"{'='*80}")
    
    # Step 1: Sample VDJ combinations using VDJ EBM
    sampled_vdj = ebm_manager.sample(num_samples=num_samples, temperature=ebm_temperature)
    
    print(f"\nSampled {len(sampled_vdj)} VDJ combinations:")
    for i, (v, d, j) in enumerate(sampled_vdj[:10]):  # Show first 10
        print(f"  {i+1}. V={v}, D={d}, J={j}")
    if len(sampled_vdj) > 10:
        print(f"  ... and {len(sampled_vdj) - 10} more")
    
    # Step 2: Generate antibody sequences for each sampled VDJ combination
    print(f"\n{'='*80}")
    print("Step 2: Generating antibody sequences using CDR3 model")
    print(f"{'='*80}")
    
    # Extract V, D, J calls
    v_calls = [vdj[0] for vdj in sampled_vdj]
    d_calls = [vdj[1] for vdj in sampled_vdj]
    j_calls = [vdj[2] for vdj in sampled_vdj]
    
    # Generate sequences using CDR3 model
    generated_sequences, _, intermediate_sequences = sample_from_vdj(
        model=cdr3_model,
        v_calls=v_calls,
        d_calls=d_calls,
        j_calls=j_calls,
        vdj_embeddings_dict=vdj_embeddings_dict,
        tokenizer=tokenizer,
        device=device,
        n_steps=cdr3_n_steps,
        temperature=cdr3_temperature,
        seed=seed,
        output_file=None,  # We'll save manually with full information
        save_at_times=save_at_times,
        track_edit_rates=False,
        random_aas_num=random_aas_num
    )
    
    # Step 3: Save results
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Get V and J sequences for output
        # Reconstruct v_end_j_start using the same random seed as sample_from_vdj
        v_sequences = []
        j_sequences = []
        v_end_j_start_list = []
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        rng = np.random.RandomState(seed if seed is not None else None)
        for v_call, j_call in zip(v_calls, j_calls):
            v_seq = germline_sequences_V[v_call]["aa_seq"]
            j_seq = germline_sequences_J[j_call]["aa_seq"]
            v_sequences.append(v_seq)
            j_sequences.append(j_seq)
            # Reconstruct v_end_j_start (same logic as in sample_from_vdj)
            random_aas = ''.join(rng.choice(amino_acids, size=random_aas_num))
            v_end_j_start_list.append(v_seq[-6:] + random_aas + j_seq[:6])
        
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
            for i, ((v_call, d_call, j_call), v_seq, j_seq, v_end_j, gen_seq) in enumerate(
                zip(sampled_vdj, v_sequences, j_sequences, v_end_j_start_list, generated_sequences)
            ):
                row = [v_call, d_call, j_call, v_seq, j_seq, v_end_j, gen_seq]
                if save_at_times is not None:
                    for t_save in sorted(save_at_times):
                        if i < len(intermediate_sequences.get(t_save, [])):
                            row.append(intermediate_sequences[t_save][i])
                        else:
                            row.append("")
                writer.writerow(row)
        
        print(f"\nCSV file saved to {csv_file}")
        
        # Save FASTA file (only generated sequences)
        fasta_file = csv_file.replace('.csv', '.fasta')
        with open(fasta_file, 'w', encoding='utf-8') as f:
            for i, ((v_call, d_call, j_call), gen_seq) in enumerate(
                zip(sampled_vdj, generated_sequences)
            ):
                # FASTA header: sample index and VDJ information
                header = f">sample_{i+1}_V_{v_call}_D_{d_call}_J_{j_call}"
                f.write(f"{header}\n{gen_seq}\n")
        
        print(f"FASTA file saved to {fasta_file}")
    
    return sampled_vdj, generated_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full Inference: VDJ EBM + CDR3 Generation')
    
    # Model checkpoints
    parser.add_argument('--ebm_checkpoint', type=str, required=True,
                        help='Path to VDJ EBM checkpoint')
    parser.add_argument('--cdr3_checkpoint', type=str, required=True,
                        help='Path to CDR3 generation model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary json file')
    parser.add_argument('--vdj_emb_dict_path', type=str, required=True,
                        help='Path to VDJ embeddings dictionary')
    
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of VDJ combinations to sample (and sequences to generate)')
    parser.add_argument('--ebm_temperature', type=float, default=1.0,
                        help='Temperature for VDJ EBM sampling')
    parser.add_argument('--cdr3_n_steps', type=int, default=1000,
                        help='Number of Euler steps for CDR3 generation')
    parser.add_argument('--cdr3_temperature', type=float, default=1.0,
                        help='Temperature for CDR3 generation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV file (will also generate a FASTA file)')
    parser.add_argument('--save_at_times', type=str, default=None,
                        help='Comma-separated list of time points to save intermediate sequences')
    parser.add_argument('--random_aas_num', type=int, default=15,
                        help='Number of random amino acids to insert between v_end and j_start')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, or cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load VDJ embeddings dictionary
    print(f"\nLoading VDJ embeddings from {args.vdj_emb_dict_path}")
    vdj_embeddings_dict = torch.load(args.vdj_emb_dict_path, map_location='cpu')
    print(f"Loaded {len(vdj_embeddings_dict)} VDJ embeddings")
    
    # Load VDJ EBM model
    print(f"\nLoading VDJ EBM model from {args.ebm_checkpoint}")
    ebm_manager = load_ebm_model(args.ebm_checkpoint, vdj_embeddings_dict, device)
    
    # Load CDR3 generation model
    print(f"\nLoading CDR3 generation model from {args.cdr3_checkpoint}")
    cdr3_model = load_cdr3_model(args.cdr3_checkpoint, device, vdj_embeddings_dict=vdj_embeddings_dict)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.vocab}")
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)
    
    # Parse save_at_times if provided
    save_at_times = None
    if args.save_at_times:
        save_at_times = [float(t.strip()) for t in args.save_at_times.split(',')]
        print(f"Will save intermediate sequences at times: {save_at_times}")
    
    # Run full inference
    sampled_vdj, generated_sequences = full_inference(
        ebm_manager=ebm_manager,
        cdr3_model=cdr3_model,
        vdj_embeddings_dict=vdj_embeddings_dict,
        tokenizer=tokenizer,
        device=device,
        num_samples=args.num_samples,
        ebm_temperature=args.ebm_temperature,
        cdr3_n_steps=args.cdr3_n_steps,
        cdr3_temperature=args.cdr3_temperature,
        seed=args.seed,
        output_file=args.output_file,
        save_at_times=save_at_times,
        random_aas_num=args.random_aas_num
    )
    
    print(f"\n{'='*80}")
    print("Full Inference Complete!")
    print(f"{'='*80}")
    print(f"Generated {len(generated_sequences)} antibody sequences")
    print("\nFirst 5 generated sequences:")
    for i, seq in enumerate(generated_sequences[:5]):
        print(f"  {i+1}. {seq[:100]}{'...' if len(seq) > 100 else ''}")

