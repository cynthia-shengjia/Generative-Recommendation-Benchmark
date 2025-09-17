import torch
from typing import List, Tuple

def asynchronous_complete_cumsum_py(lengths: torch.Tensor) -> torch.Tensor:
    """
    功能等价于 torch.ops.fbgemm.asynchronous_complete_cumsum
    根据序列长度计算偏移量。
    
    Args:
        lengths (torch.Tensor): 形如 [len_1, len_2, ...] 的一维张量。

    Returns:
        torch.Tensor: 形如 [0, len_1, len_1+len_2, ...] 的偏移量张量。
    """
    zero = torch.tensor([0], device=lengths.device, dtype=torch.int64)
    return torch.cat([zero, torch.cumsum(lengths, dim=0, dtype=torch.int64)])

def jagged_to_padded_dense_py(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    jagged_to_padded_dense 的 PyTorch 实现，支持左Padding。
    """
    offset_tensor = offsets[0]
    max_length = max_lengths[0]
    batch_size = len(offset_tensor) - 1

    if batch_size == 0:
        shape = (0, max_length) if values.dim() == 1 else (0, max_length, values.size(1))
        return torch.empty(shape, dtype=values.dtype, device=values.device)

    shape = (batch_size, max_length) if values.dim() == 1 else (batch_size, max_length, values.size(1))
    padded_dense = torch.full(shape, padding_value, dtype=values.dtype, device=values.device)
    lengths = offset_tensor[1:] - offset_tensor[:-1]

    indices = torch.arange(max_length, device=values.device)
    start_indices = max_length - lengths.unsqueeze(1) 

    mask = indices >= start_indices

    padded_dense[mask] = values
    
    return padded_dense


def dense_to_jagged_py(
    dense: torch.Tensor, 
    offsets: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:

    offset_tensor = offsets[0]
    batch_size = len(offset_tensor) - 1
    max_length = dense.size(1)

    if batch_size == 0:
        shape = (0,) if dense.dim() == 2 else (0, dense.size(2))
        return (torch.empty(shape, dtype=dense.dtype, device=dense.device), offsets)

    lengths = offset_tensor[1:] - offset_tensor[:-1]
    indices = torch.arange(max_length, device=dense.device)
    start_indices = max_length - lengths.unsqueeze(1)
    mask = indices >= start_indices
    values = dense[mask]
    
    return (values, offsets)