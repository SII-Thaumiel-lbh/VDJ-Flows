import torch
import torch.nn.functional as F
from typing import Tuple
from torchtyping import TensorType as T
from typing import cast


def rm_gap_tokens(z: torch.Tensor, pad_token: int, gap_token: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove gap tokens from a batched tensor and right-pad with pad_token.
    
    Args:
        z: Tensor with gap tokens
        pad_token: Token ID for padding
        gap_token: Token ID for gaps
    """    
    batch_size, _ = z.shape
    z_no_gap = []
    for b in range(batch_size):
        z_no_pad = z[b][z[b] != pad_token]
        z_no_gap.append(z_no_pad[z_no_pad != gap_token])
    max_len = max(len(z) for z in z_no_gap)
    x = torch.stack([F.pad(z, (0, max_len - len(z)), value=pad_token) for z in z_no_gap], dim=0).long()
    x_pad_mask = (x == pad_token)
    z_gap_mask = (z == gap_token)
    z_pad_mask = (z == pad_token)
    assert ((~x_pad_mask).sum(1) + z_gap_mask.sum(1)).equal((~z_pad_mask).sum(1))
    return x, x_pad_mask, z_gap_mask, z_pad_mask


def rv_gap_tokens(x: torch.Tensor, z_gap_mask: torch.Tensor, z_pad_mask: torch.Tensor,
                  pad_token: int, gap_token: int) -> torch.Tensor:
    """
    Reinsert gap tokens into a tensor at specified positions.
    
    Args:
        x: Tensor without gap tokens
        z_gap_mask: Mask indicating gap positions
        z_pad_mask: Mask indicating pad positions
        pad_token: Token ID for padding
        gap_token: Token ID for gaps
    """
    assert x.shape[0] == z_gap_mask.shape[0]
    assert x.shape[1] <= z_gap_mask.shape[1]
    assert z_gap_mask.shape == z_pad_mask.shape
    batch_size, _ = x.shape
    _, z_seq_len = z_gap_mask.shape
    z = torch.full((batch_size, z_seq_len), pad_token, dtype=x.dtype, device=x.device)    
    z[~z_gap_mask & ~z_pad_mask] = x[x != pad_token]
    z[z_gap_mask] = gap_token
    return z


def fill_gap_tokens_with_repeats(
    x_ut: torch.Tensor,
    z_gap_mask: torch.Tensor,
    z_pad_mask: torch.Tensor,
):
    """
    Map edit rates from X space back to Z space.
    For GAP positions, use the edit rates from the corresponding non-GAP position.
    """
    batch_size, _ = z_gap_mask.shape
    _, x_seq_len, _ = x_ut.shape

    # Use cumsum on non-gap positions to point to the last valid non-gap position
    non_gap_mask = ~z_gap_mask  # Invert mask to get non-gap positions
    indices = non_gap_mask.cumsum(dim=1) - 1        # (batch_size, z_seq_len)
    indices = indices.clamp(min=0, max=x_seq_len-1) # Ensure indices are within bounds

    # Use indices to gather from x_ut
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]   # (batch_size, z_seq_len, vocab_size)
    result[z_pad_mask] = 0                  # Set pad positions to 0
    return result

def make_ut_mask_from_z(
    z_t: T["batch_size", "z_seq_len", "long"],
    z_1: T["batch_size", "z_seq_len", "long"],
    vocab_size: int,
    pad_token: int,
    gap_token: int,
) -> T["batch_size", "z_seq_len", "n_ops", "bool"]:
    """
    Create a mask for u_cat for indexing the output rate tensor based on differences between z_t and z_1.
    For each position i where z_t and z_1 differ, we index as follows:

    - z_t[i] = GAP_TOKEN & z_1[i] = c => u_mask[i, insert, c] = 1
    - z_t[i] = c & z_1[i] = GAP_TOKEN => u_mask[i, delete] = 1
    - z_t[i] = c1 & z_1[i] = c2 => u_mask[i, substitute, c1, c2] = 1
    """
    batch_size, z_seq_len = z_t.shape
    n_ops = 2 * vocab_size + 1  # insert + substitute + delete

    z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq         # (batch_size, z_seq_len)
    z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq         # (batch_size, z_seq_len)
    z_sub = z_neq & ~z_ins & ~z_del                                 # (batch_size, z_seq_len) 

    # mask (batch_size, z_seq_len, u_ops) where 1 indicates operation that bring z_t closer to z_1
    u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    u_mask[z_ins, z_1[z_ins]] = True
    u_mask[z_sub, z_1[z_sub] + vocab_size] = True
    u_mask[:,:,-1][z_del] = True

    assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
    assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

    return cast(T["batch_size", "z_seq_len", "n_ops", "bool"], u_mask)
