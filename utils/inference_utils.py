import torch
from typing import Optional

def pretty_print(x: torch.Tensor, tokenizer=None, return_str=False, **kwargs) -> Optional[str]:
    """
    Pretty print a tensor as an ascii string with gap tokens represented as 'Δ'
    Non-printable/special characters (including line breaks, tabs, etc.) are replaced with '.'
    
    This function combines the functionality of safe_chr and pretty_print.
    Use return_str=True to get the string representation instead of printing.
    
    Args:
        x: Tensor of token IDs
        tokenizer: Custom tokenizer for decoding (required)
        return_str: If True, return the string instead of printing (default: False)
        **kwargs: Additional arguments:
            - compact: Whether to use compact representation (default: False)
            - show_special_chars: Whether to show special characters (default: False, currently unused)
    
    Returns:
        If return_str=True, returns the string representation. Otherwise, prints and returns None.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for pretty_print")
    
    def _safe_chr(c: int) -> str:
        """Internal helper to convert token ID to character."""
        pad_token = tokenizer.pad_token_id
        bos_token = tokenizer.bos_token_id
        gap_token = tokenizer.gap_token_id
        compact = kwargs.get('compact', False)
        
        if c == gap_token:
            return 'Δ' if compact else '<GAP>'
        elif c == pad_token:
            return 'π' if compact else '<PAD>'
        elif c == bos_token:
            return '<BOS>'
        
        return tokenizer.id_to_token[c] if c in tokenizer.id_to_token else '.'
    
    # Convert tensor to string
    x_str = ''.join(_safe_chr(int(c)) for c in x.cpu().numpy().flatten())
    
    if return_str:
        return x_str
    else:
        print(x_str)
        return None