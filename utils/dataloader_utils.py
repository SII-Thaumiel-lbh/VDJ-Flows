import torch
import json
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from utils.flow import Coupling, EmptyCoupling
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


def _align_pair_numpy(seq_0: np.ndarray, seq_1: np.ndarray, gap_token: int) -> Tuple[List[int], List[int]]:
    """
    Optimized alignment using numpy arrays instead of Python lists for DP table.
    This is significantly faster than the original list-based implementation.
    
    Args:
        seq_0: First sequence as numpy array
        seq_1: Second sequence as numpy array
        gap_token: Token ID to use for gaps
    """
    m, n = len(seq_0), len(seq_1)
    
    # Use numpy array for DP table - much faster than nested lists
    # Pre-allocate with proper initialization
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[0, :] = np.arange(n + 1, dtype=np.int32)
    dp[:, 0] = np.arange(m + 1, dtype=np.int32)
    
    # Fill DP table using vectorized operations where possible
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_0[i-1] == seq_1[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    
    # Backtrack (Python list operations, not easily JIT-compiled)
    aligned_0, aligned_1 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_0[i-1] == seq_1[j-1]:
            aligned_0.append(int(seq_0[i-1]))
            aligned_1.append(int(seq_1[j-1]))
            i, j = i-1, j-1
        elif i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + 1:
            aligned_0.append(int(seq_0[i-1]))
            aligned_1.append(int(seq_1[j-1]))
            i, j = i-1, j-1
        elif i > 0 and dp[i, j] == dp[i-1, j] + 1:
            aligned_0.append(int(seq_0[i-1]))
            aligned_1.append(gap_token)
            i -= 1
        else:
            aligned_0.append(gap_token)
            aligned_1.append(int(seq_1[j-1]))
            j -= 1
    
    return aligned_0[::-1], aligned_1[::-1]


def align_pair(seq_0: List[int], seq_1: List[int], gap_token: int) -> Tuple[List[int], List[int]]:
    """
    Aligns two sequences using dynamic programming to find the minimum edit distance.
    Returns two lists representing the aligned sequences.
    
    Optimized version using numpy arrays for better performance.
    
    Args:
        seq_0: First sequence
        seq_1: Second sequence
        gap_token: Token ID to use for gaps
    """
    # Convert to numpy arrays if not already
    if isinstance(seq_0, torch.Tensor):
        seq_0 = seq_0.cpu().numpy()
    if isinstance(seq_1, torch.Tensor):
        seq_1 = seq_1.cpu().numpy()
    
    if not isinstance(seq_0, np.ndarray):
        seq_0 = np.array(seq_0, dtype=np.int32)
    if not isinstance(seq_1, np.ndarray):
        seq_1 = np.array(seq_1, dtype=np.int32)
    
    return _align_pair_numpy(seq_0, seq_1, gap_token)

def _align_single_pair(args):
    """Helper function for parallel alignment"""
    x_0_seq, x_1_seq, gap_token = args
    return align_pair(x_0_seq, x_1_seq, gap_token)


def opt_align_xs_to_zs(
    x_0: torch.Tensor, 
    x_1: torch.Tensor, 
    gap_token: int,
    use_parallel: bool = True,
    max_workers: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aligns x_0 and x_1 to the same length by using a dynamic programming approach
    to find the minimum edit distance alignment.
    
    Optimized with parallel processing for batch alignment.
    
    Args:
        x_0: First sequence tensor (batch_size, seq_len)
        x_1: Second sequence tensor (batch_size, seq_len)
        gap_token: Token ID to use for gaps
        use_parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of worker threads/processes (None = auto)
    """
    batch_size = x_0.shape[0]
    
    # Convert to CPU and numpy for faster processing
    x_0_cpu = x_0.cpu() if x_0.is_cuda else x_0
    x_1_cpu = x_1.cpu() if x_1.is_cuda else x_1
    
    if use_parallel and batch_size > 1:
        # Use parallel processing for batch alignment
        if max_workers is None:
            # Use number of CPU cores, but limit to batch size
            max_workers = min(mp.cpu_count(), batch_size, 8)  # Cap at 8 to avoid overhead
        
        # Prepare arguments for parallel processing
        align_args = [
            (x_0_cpu[b].numpy(), x_1_cpu[b].numpy(), gap_token)
            for b in range(batch_size)
        ]
        
        # Use ThreadPoolExecutor for I/O-bound tasks (numpy operations are GIL-friendly)
        # For CPU-intensive tasks, ProcessPoolExecutor might be better but has more overhead
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            aligned_pairs = list(executor.map(_align_single_pair, align_args))
    else:
        # Sequential processing (fallback or for small batches)
        aligned_pairs = [
            align_pair(x_0_cpu[b].numpy(), x_1_cpu[b].numpy(), gap_token)
            for b in range(batch_size)
        ]
    
    # Stack results back into tensors
    x_0_aligned = torch.stack([
        torch.tensor(pair[0], dtype=x_0.dtype, device=x_0.device)
        for pair in aligned_pairs
    ])
    x_1_aligned = torch.stack([
        torch.tensor(pair[1], dtype=x_1.dtype, device=x_1.device)
        for pair in aligned_pairs
    ])
    
    return x_0_aligned, x_1_aligned


# def naive_align_xs_to_zs(x_0: torch.Tensor, x_1: torch.Tensor, gap_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Aligns x_0 and x_1 to the same length by padding with gap_token.
    
#     Args:
#         x_0: First sequence tensor
#         x_1: Second sequence tensor
#         gap_token: Token ID to use for gaps
#     """
#     max_len = max(x_0.shape[1], x_1.shape[1])
#     x_0_padded = F.pad(x_0, (0, max_len - x_0.shape[1]), value=gap_token)
#     x_1_padded = F.pad(x_1, (0, max_len - x_1.shape[1]), value=gap_token)
#     return x_0_padded, x_1_padded


# def shifted_align_xs_to_zs(x_0: torch.Tensor, x_1: torch.Tensor, 
#                            gap_token: int, pad_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Aligns x_0 and z_1 by shifting x_1 to the right by the length of x_0, then
#     padding all sequences to the same length with gap tokens.
    
#     Args:
#         x_0: First sequence tensor
#         x_1: Second sequence tensor
#         gap_token: Token ID to use for gaps
#         pad_token: Token ID to use for padding
#     """
#     batch_size, _ = x_0.shape
#     x0_seq_lens = (~(x_0 == gap_token)).sum(dim=1)
#     x1_seq_lens = (~(x_1 == gap_token)).sum(dim=1)
#     z_seq_lens = x0_seq_lens + x1_seq_lens
#     max_z_len = int(z_seq_lens.max().item())
#     z_0 = torch.full((batch_size, max_z_len), gap_token, dtype=x_0.dtype, device=x_0.device)
#     z_1 = torch.full((batch_size, max_z_len), gap_token, dtype=x_1.dtype, device=x_1.device)
#     batch_indices = torch.arange(batch_size, device=x_0.device).unsqueeze(1)
#     z_0[batch_indices, :x0_seq_lens] = x_0
#     z_1[batch_indices, x0_seq_lens:] = x_1
#     z_0[batch_indices, z_seq_lens:] = pad_token
#     z_1[batch_indices, z_seq_lens:] = pad_token
#     return z_0, z_1

def parse_sample_group(sample_group):
    """parse sample group, return JSON data list (for unlisted flatten)"""
    json_datas = []
    json_keys = [key for key in sample_group.keys() if key.endswith('.json')]
    for json_key in json_keys:
        try:
            json_data = json.loads(sample_group[json_key].decode("utf-8"))
            json_datas.append(json_data)
        except (json.JSONDecodeError, KeyError):
            continue
    return json_datas