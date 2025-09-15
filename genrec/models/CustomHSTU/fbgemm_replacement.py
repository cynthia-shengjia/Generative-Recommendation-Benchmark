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


# ==============================================================================
# ======================== 优化的版本 =========================================
# ==============================================================================
def jagged_to_padded_dense_py(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    jagged_to_padded_dense 的向量化、高性能 PyTorch 实现。
    """
    offset_tensor = offsets[0]
    max_length = max_lengths[0]
    batch_size = len(offset_tensor) - 1

    if batch_size == 0:
        shape = (0, max_length)
        if values.dim() > 1:
            shape = (0, max_length, values.size(1))
        return torch.empty(shape, dtype=values.dtype, device=values.device)

    # 1. 创建一个填满 padding_value 的目标张量 (与原版相同)
    shape = (batch_size, max_length)
    if values.dim() > 1:
        shape = (batch_size, max_length, values.size(1))
    padded_dense = torch.full(shape, padding_value, dtype=values.dtype, device=values.device)
    
    # 2. 计算每个序列的真实长度
    # lengths 形如 [len_1, len_2, ...]
    lengths = offset_tensor[1:] - offset_tensor[:-1]

    # 3. 创建一个布尔掩码 (boolean mask) 来标识所有有效元素的位置
    # torch.arange(max_length) -> [0, 1, 2, ..., max_length-1]
    # lengths.unsqueeze(1) -> [[len_1], [len_2], ...]
    # 通过广播机制，mask 是一个 (batch_size, max_length) 的布尔张量
    # 对于第 i 行，前 lengths[i] 个元素为 True，其余为 False
    mask = torch.arange(max_length, device=values.device) < lengths.unsqueeze(1)
    
    # 4. 使用掩码直接进行赋值 (一次性、并行的操作)
    # padded_dense[mask] 会返回一个扁平化的一维视图，其元素数量正好等于 values 的数量
    # 然后我们将 values 的所有元素一次性地“填充”到这些为 True 的位置
    padded_dense[mask] = values
    
    return padded_dense


def dense_to_jagged_py(
    dense: torch.Tensor, 
    offsets: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    dense_to_jagged 的向量化、高性能 PyTorch 实现。
    """
    offset_tensor = offsets[0]
    batch_size = len(offset_tensor) - 1
    max_length = dense.size(1)

    if batch_size == 0:
        shape = (0,)
        if dense.dim() > 2:
            shape = (0, dense.size(2))
        return (torch.empty(shape, dtype=dense.dtype, device=dense.device), offsets)

    # 1. 同样，计算每个序列的真实长度
    lengths = offset_tensor[1:] - offset_tensor[:-1]

    # 2. 创建与上面完全相同的布尔掩码
    mask = torch.arange(max_length, device=dense.device) < lengths.unsqueeze(1)

    # 3. 使用掩码直接从 dense 张量中选取所有有效元素
    # dense[mask] 会返回一个 (total_valid_elements, dim) 的张量，这正是我们想要的 values
    values = dense[mask]
    
    return (values, offsets)