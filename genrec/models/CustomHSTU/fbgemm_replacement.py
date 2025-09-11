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
    # FBGEMM 通常返回 int32 或 int64，这里我们根据 PyTorch 推荐使用 int64
    # 并且确保偏移量张量与长度张量在同一设备上
    zero = torch.tensor([0], device=lengths.device, dtype=torch.int64)
    return torch.cat([zero, torch.cumsum(lengths, dim=0, dtype=torch.int64)])


def jagged_to_padded_dense_py(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    功能等价于 torch.ops.fbgemm.jagged_to_padded_dense
    将 Jagged Tensor (values + offsets) 转换为填充后的 Dense Tensor。
    
    Args:
        values (torch.Tensor): 包含所有序列数据的1D或2D张量 (sum_lengths, dim)。
        offsets (List[torch.Tensor]): 包含偏移量张量的列表，我们只处理第一个。
        max_lengths (List[int]): 包含最大长度的列表，我们只处理第一个。
        padding_value (float): 用于填充的值。

    Returns:
        torch.Tensor: 填充后的2D或3D张量 (batch_size, max_length, dim)。
    """
    offset_tensor = offsets[0]
    max_length = max_lengths[0]
    
    batch_size = len(offset_tensor) - 1
    if batch_size == 0:
        # 处理空批次的情况
        shape = (0, max_length)
        if values.dim() > 1:
            shape = (0, max_length, values.size(1))
        return torch.empty(shape, dtype=values.dtype, device=values.device)

    # 创建一个填满 padding_value 的目标张量
    shape = (batch_size, max_length)
    if values.dim() > 1:
        # 如果 values 是 (N, D)，输出就是 (B, L, D)
        shape = (batch_size, max_length, values.size(1))

    padded_dense = torch.full(shape, padding_value, dtype=values.dtype, device=values.device)

    # 循环遍历每个序列并填充到目标张量中
    for i in range(batch_size):
        start = offset_tensor[i]
        end = offset_tensor[i+1]
        length = end - start
        if length > 0:
            padded_dense[i, :length] = values[start:end]
            
    return padded_dense


def dense_to_jagged_py(
    dense: torch.Tensor, 
    offsets: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    功能等价于 torch.ops.fbgemm.dense_to_jagged
    将填充后的 Dense Tensor 转换为 Jagged Tensor (values)。
    
    Args:
        dense (torch.Tensor): 填充后的2D或3D张量 (batch_size, max_length, ...)。
        offsets (List[torch.Tensor]): 包含偏移量张量的列表，我们只处理第一个。

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: 返回 values 张量和原始的 offsets 列表。
    """
    offset_tensor = offsets[0]
    batch_size = len(offset_tensor) - 1
    
    if batch_size == 0:
        shape = (0,)
        if dense.dim() > 2:
            shape = (0, dense.size(2))
        return (torch.empty(shape, dtype=dense.dtype, device=dense.device), offsets)

    sequences = []
    for i in range(batch_size):
        length = offset_tensor[i+1] - offset_tensor[i]
        if length > 0:
            sequences.append(dense[i, :length])

    # 如果所有序列长度都为0，则返回一个空的 values 张量
    if not sequences:
        shape = (0,)
        if dense.dim() > 2:
            shape = (0, dense.size(2))
        return (torch.empty(shape, dtype=dense.dtype, device=dense.device), offsets)
        
    values = torch.cat(sequences, dim=0)
    
    # fbgemm 操作返回一个元组，第一个元素是 values
    return (values, offsets)