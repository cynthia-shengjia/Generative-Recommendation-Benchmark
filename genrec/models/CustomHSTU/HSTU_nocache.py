# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152).
"""

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import logging

from .hstu_utils4decoderOnly import (
    EmbeddingModule,
    InputFeaturesPreprocessorModule,
    OutputPostprocessorModule,
    get_current_embeddings,
    GeneralizedInteractionModule,
    NDPModule
)
from .fbgemm_replacement import (
    asynchronous_complete_cumsum_py,
    jagged_to_padded_dense_py,
    dense_to_jagged_py,
)

TIMESTAMPS_KEY = "timestamps"

def debug_print(tensor, name="tensor", step_counter=0):
    """一个详细的调试打印函数"""
    if tensor is None:
        print(f"[DEBUG @ step {step_counter}] {name}: None")
        return
    # 确保在CPU上打印，避免同步问题
    tensor_cpu = tensor.detach().to('cpu', non_blocking=True)
    print(
        f"[DEBUG @ step {step_counter}] {name}:\n"
        f"  - Shape: {tensor.shape}\n"
        f"  - Dtype: {tensor.dtype}\n"
        f"  - Device: {tensor.device}\n"
        f"  - Stats: min={tensor_cpu.min():.4f}, max={tensor_cpu.max():.4f}, "
        f"mean={tensor_cpu.mean():.4f}, std={tensor_cpu.std():.4f}\n"
        f"  - Has NaN: {torch.isnan(tensor_cpu).any()}\n"
        f"  - Has Inf: {torch.isinf(tensor_cpu).any()}"
    )
class RelativeAttentionBiasModule(torch.nn.Module):

    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = all_timestamps.size(1)
        # N = self._max_seq_len
        return torch.zeros(B, N, N, device=self._pos_w.device, dtype=self._pos_w.dtype)
        # t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        # t = t[..., :-N].reshape(1, N, 3 * N - 2)
        # r = (2 * N - 1) // 2
        # # [B, N + 1] to simplify tensor manipulations.
        # ext_timestamps = torch.cat(
        #     [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
修复的HSTU实现，支持真正的Cache机制
"""

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import logging

from .hstu_utils4decoderOnly import (
    EmbeddingModule,
    InputFeaturesPreprocessorModule,
    OutputPostprocessorModule,
    get_current_embeddings,
    GeneralizedInteractionModule,
    NDPModule
)
from .fbgemm_replacement import (
    asynchronous_complete_cumsum_py,
    jagged_to_padded_dense_py,
    dense_to_jagged_py,
)

TIMESTAMPS_KEY = "timestamps"

def debug_print(tensor, name="tensor", step_counter=0):
    """详细的调试打印函数"""
    if tensor is None:
        print(f"[DEBUG @ step {step_counter}] {name}: None")
        return
    tensor_cpu = tensor.detach().to('cpu', non_blocking=True)
    print(
        f"[DEBUG @ step {step_counter}] {name}:\n"
        f"  - Shape: {tensor.shape}\n"
        f"  - Dtype: {tensor.dtype}\n"
        f"  - Device: {tensor.device}\n"
        f"  - Stats: min={tensor_cpu.min():.4f}, max={tensor_cpu.max():.4f}, "
        f"mean={tensor_cpu.mean():.4f}, std={tensor_cpu.std():.4f}\n"
        f"  - Has NaN: {torch.isnan(tensor_cpu).any()}\n"
        f"  - Has Inf: {torch.isinf(tensor_cpu).any()}"
    )

class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass

class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]

class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    修复的相对时间和位置偏置模块，支持真正的偏置计算
    """
    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )
    
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        修复的前向传播，实际计算相对偏置而不是返回零矩阵
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = all_timestamps.size(1)
        
        # 如果没有时间戳，只使用位置偏置
        if all_timestamps is None:
            return self._compute_position_bias_only(B, N)
        
        # 计算位置偏置
        pos_bias = self._compute_position_bias(N)
        
        # 计算时间偏置
        time_bias = self._compute_time_bias(all_timestamps)
        
        return pos_bias + time_bias
    
    def _compute_position_bias_only(self, B: int, N: int) -> torch.Tensor:
        """仅计算位置偏置"""
        pos_bias = self._compute_position_bias(N)
        return pos_bias.expand(B, -1, -1)
    
    def _compute_position_bias(self, N: int) -> torch.Tensor:
        """计算位置偏置"""
        if N > self._max_seq_len:
            # 如果序列长度超过最大长度，截断或填充
            N = min(N, self._max_seq_len)
        
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2
        return t[..., r:-r]
    
    def _compute_time_bias(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        """计算时间偏置"""
        B, N = all_timestamps.shape
        
        # [B, N + 1] 为了简化张量操作
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        
        # 计算时间差矩阵 (B, N, N)
        time_diffs = ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
        
        # 应用因果掩码
        causal_mask = torch.tril(torch.ones(N, N, device=all_timestamps.device, dtype=torch.bool))
        time_diffs = time_diffs * causal_mask.unsqueeze(0).float()
        
        # 分桶化时间差
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(time_diffs),
            min=0,
            max=self._num_buckets,
        ).detach()
        
        # 查找对应的时间偏置
        time_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        
        return time_bias

# Cache状态定义: (v, padded_q, padded_k, new_outputs)
HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_offsets: torch.Tensor,
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: RelativeAttentionBiasModule,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    
    # 修复的缓存处理逻辑
    if delta_x_offsets is not None and cached_q is not None and cached_k is not None:
        # 增量更新模式 - 使用缓存
        padded_q, padded_k = cached_q.clone(), cached_k.clone()
        
        # 计算新token在padded tensor中的位置
        flattened_offsets = delta_x_offsets[1] + torch.arange(
            start=0,
            end=B * n,
            step=n,
            device=delta_x_offsets[1].device,
            dtype=delta_x_offsets[1].dtype,
        )
        
        # 更新padded_q和padded_k
        padded_q = (
            padded_q.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=q,
            )
            .view(B, n, -1)
        )
        padded_k = (
            padded_k.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=k,
            )
            .view(B, n, -1)
        )
    else:
        # 常规模式 - 完整计算
        padded_q = jagged_to_padded_dense_py(
            values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )
        padded_k = jagged_to_padded_dense_py(
            values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )

    # 计算注意力分数
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q.view(B, n, num_heads, attention_dim),
        padded_k.view(B, n, num_heads, attention_dim),
    )
    
    # 添加相对偏置
    if all_timestamps is not None and rel_attn_bias is not None:
        rel_bias = rel_attn_bias(all_timestamps)
        qk_attn = qk_attn + rel_bias.unsqueeze(1)
    
    # 应用激活函数和归一化
    qk_attn = F.silu(qk_attn) / n
    
    # 应用注意力掩码
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(1)
    
    # 计算注意力输出
    padded_v = jagged_to_padded_dense_py(v, [x_offsets], [n])
    attn_output = dense_to_jagged_py(
        torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            padded_v.reshape(B, n, num_heads, linear_dim),
        ).reshape(B, n, num_heads * linear_dim),
        [x_offsets],
    )[0]
    
    return attn_output, padded_q, padded_k

class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
            
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[HSTUCacheState] = None,
        return_cache_states: bool = False,
    ):
        """
        修复的前向传播，包含完整的缓存处理逻辑
        """
        n: int = invalid_attn_mask.size(-1)
        B: int = x_offsets.size(0) - 1
        
        # 缓存处理
        cached_v = None
        cached_q = None
        cached_k = None
        cached_outputs = None
        
        if cache is not None:
            cached_v, cached_q, cached_k, cached_outputs = cache
        
        # 对于增量生成，限制x到新token
        if delta_x_offsets is not None and cached_outputs is not None:
            x = x[delta_x_offsets[0], :]
        
        # 输入归一化
        normed_x = self._norm_input(x)

        # 线性变换
        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                pass
            else:
                raise ValueError(f"Unknown activation {self._linear_activation}")
                
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        # 更新v缓存
        if delta_x_offsets is not None and cached_v is not None:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        # 注意力计算
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            attn_output, padded_q, padded_k = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
            )
        else:
            raise ValueError(f"Unsupported normalization method {self._normalization}")

        # 对于增量生成，只取新的输出
        if delta_x_offsets is not None:
            attn_output = attn_output[delta_x_offsets[0], :]

        # 最终变换
        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        block_output = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )
        
        # 更新输出缓存
        new_outputs = block_output
        if delta_x_offsets is not None and cached_outputs is not None:
            new_outputs = cached_outputs.index_copy_(
                dim=0, index=delta_x_offsets[0], source=new_outputs
            )

        # 准备新的缓存状态
        new_cache_state = None
        if return_cache_states:
            new_cache_state = (v.contiguous(), padded_q, padded_k, new_outputs)

        return new_outputs, new_cache_state

class HSTUJagged(torch.nn.Module):
    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()
        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """修复的jagged前向传播"""
        cache_states: List[HSTUCacheState] = []

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                layer_cache = cache[i] if cache is not None else None
                
                x, cache_states_i = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=layer_cache,
                    return_cache_states=return_cache_states,
                )

                if return_cache_states and cache_states_i is not None:
                    cache_states.append(cache_states_i)

        return x, cache_states

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """前向传播入口"""
        if len(x.size()) == 3:
            x = dense_to_jagged_py(x, [x_offsets])[0]

        jagged_x, cache_states = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        
        y = jagged_to_padded_dense_py(
            values=jagged_x,
            offsets=[x_offsets],
            max_lengths=[invalid_attn_mask.size(1)],
            padding_value=0.0,
        )
        return y, cache_states

class HSTU(GeneralizedInteractionModule):
    """
    修复的HSTU实现，支持真正的缓存机制
    """
    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)
        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len + max_output_len,
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        
        # 因果掩码
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (self._max_sequence_length, self._max_sequence_length),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(f"Initialize {name} as xavier normal: {params.data.size()} params")
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[HSTUCacheState]]]:
        """
        修复的用户embedding生成，包含完整的缓存支持
        """
        device = past_lengths.device
        B, N, _ = past_embeddings.size()

        # 输入预处理
        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        # 构建注意力掩码
        float_dtype = user_embeddings.dtype
        causal_mask = torch.tril(torch.ones((N, N), device=device, dtype=torch.bool))
        padding_mask = torch.arange(N, device=device) >= (N - past_lengths.unsqueeze(1))
        combined_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        attention_mask = causal_mask.unsqueeze(0) * combined_padding_mask
        final_attention_mask = attention_mask.to(user_embeddings.dtype)

        # 调用HSTU核心模块
        user_embeddings, cached_states = self._hstu(
            x=user_embeddings,
            x_offsets=asynchronous_complete_cumsum_py(past_lengths),
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            invalid_attn_mask=final_attention_mask,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )

        # 输出后处理
        final_embeddings = self._output_postproc(user_embeddings)

        return final_embeddings, cached_states if return_cache_states else None

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        主要编码器入口
        """
        encoded_embeddings, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return encoded_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        编码当前状态的隐藏表示
        """
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        
        current_embeddings = get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )
        
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings
        # )
        # # causal masking. Otherwise [:, :-1] - [:, 1:] works
        # bucketed_timestamps = torch.clamp(
        #     self._bucketization_fn(
        #         ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
        #     ),
        #     min=0,
        #     max=self._num_buckets,
        # ).detach()
        # rel_pos_bias = t[:, :, r:-r]
        # rel_ts_bias = torch.index_select(
        #     self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        # ).view(B, N, N)
        # #禁用时间偏置和相对位置偏置，会不会好一些
        # return rel_pos_bias + rel_ts_bias


#  (v, padded_q, padded_k, new_outputs)
HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

#TODO：看一下forward中，到底是因为什么报错
def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_offsets: torch.Tensor,
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: RelativeAttentionBiasModule,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    if delta_x_offsets is not None:
        padded_q, padded_k = cached_q, cached_k
        flattened_offsets = delta_x_offsets[1] + torch.arange(
            start=0,
            end=B * n,
            step=n,
            device=delta_x_offsets[1].device,
            dtype=delta_x_offsets[1].dtype,
        )
        assert isinstance(padded_q, torch.Tensor)
        assert isinstance(padded_k, torch.Tensor)
        padded_q = (
            padded_q.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=q,
            )
            .view(B, n, -1)
        )
        padded_k = (
            padded_k.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=k,
            )
            .view(B, n, -1)
        )
    else:
        padded_q = jagged_to_padded_dense_py(
            values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )
        padded_k = jagged_to_padded_dense_py(
            values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q.view(B, n, num_heads, attention_dim),
        padded_k.view(B, n, num_heads, attention_dim),
    )
    if all_timestamps is not None:
        temp = rel_attn_bias(all_timestamps)
        # print("all_timestamps:",all_timestamps.shape)
        temp = temp.unsqueeze(1)
        # print("unsqueeze temp:",temp.shape)
        # print("qk_attn",qk_attn.shape)
        qk_attn = qk_attn + temp
    qk_attn = F.silu(qk_attn) / n
    # qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(1)
    attn_output = dense_to_jagged_py(
        torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            jagged_to_padded_dense_py(v, [x_offsets], [n]).reshape(
                B, n, num_heads, linear_dim
            ),
        ).reshape(B, n, num_heads * linear_dim),
        [x_offsets],
    )[0]
    return attn_output, padded_q, padded_k
class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[HSTUCacheState] = None,
        return_cache_states: bool = False,
    ):
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """
        n: int = invalid_attn_mask.size(-1)
        cached_q = None
        cached_k = None
        if delta_x_offsets is not None:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # [delta_x_offsets[0], :].
            assert cache is not None
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache
        
        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        if delta_x_offsets is not None:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        B: int = x_offsets.size(0) - 1
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            assert self._rel_attn_bias is not None
            attn_output, padded_q, padded_k = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
            )
        elif self._normalization == "softmax_rel_bias":
            if delta_x_offsets is not None:
                B = x_offsets.size(0) - 1
                padded_q, padded_k = cached_q, cached_k
                flattened_offsets = delta_x_offsets[1] + torch.arange(
                    start=0,
                    end=B * n,
                    step=n,
                    device=delta_x_offsets[1].device,
                    dtype=delta_x_offsets[1].dtype,
                )
                assert padded_q is not None
                assert padded_k is not None
                padded_q = (
                    padded_q.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=q,
                    )
                    .view(B, n, -1)
                )
                padded_k = (
                    padded_k.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=k,
                    )
                    .view(B, n, -1)
                )
            else:
                padded_q = jagged_to_padded_dense_py(
                    values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
                padded_k = jagged_to_padded_dense_py(
                    values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )

            qk_attn = torch.einsum("bnd,bmd->bnm", padded_q, padded_k)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps)
            additive_mask = (1.0 - invalid_attn_mask) * -1e9
            qk_attn = qk_attn + additive_mask
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)
            qk_attn = qk_attn * invalid_attn_mask
            attn_output = dense_to_jagged_py(
                torch.bmm(
                    qk_attn,
                    jagged_to_padded_dense_py(v, [x_offsets], [n]),
                ),
                [x_offsets],
            )[0]
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        attn_output = (
            attn_output
            if delta_x_offsets is None
            else attn_output[delta_x_offsets[0], :]
        )
        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        block_output = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )
        
        new_outputs = block_output
        
        if delta_x_offsets is not None:
            new_outputs = cached_outputs.index_copy_(
                dim=0, index=delta_x_offsets[0], source=new_outputs
            )

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        return new_outputs, (v, padded_q, padded_k, new_outputs)
class HSTUJagged(torch.nn.Module):

    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (\sum_i N_i, D) x float
            x_offsets: (B + 1) x int32
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}
            return_cache_states: bool. True if we should return cache states.

        Returns:
            x' = f(x), (\sum_i N_i, D) x float
        """
        cache_states: List[HSTUCacheState] = []


        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                # 将当前层的输入保存下来，方便对比
                input_to_this_layer = x

                x, cache_states_i = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=cache[i] if cache is not None else None,
                    return_cache_states=return_cache_states,
                )


                if return_cache_states:
                    cache_states.append(cache_states_i)

        return x, cache_states
    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        if len(x.size()) == 3:
            x = dense_to_jagged_py(x, [x_offsets])[0]

        jagged_x, cache_states = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        y = jagged_to_padded_dense_py(
            values=jagged_x,
            offsets=[x_offsets],
            max_lengths=[invalid_attn_mask.size(1)],
            padding_value=0.0,
        )
        return y, cache_states

def debug_last_timestep(full_tensor, lengths, name="tensor", step_counter=0):
    """
    新的调试函数：只检查每个序列最后一个时间步的状态
    """
    if full_tensor.dim() != 3:
        print(f"[DEBUG @ step {step_counter}] {name}: Error - Tensor is not 3D")
        return

    # 根据 lengths 提取最后一个时间步的 hidden state
    # PyTorch 中，我们需要创建一个索引来从每个批次中选择正确的序列元素
    batch_size = full_tensor.shape[0]
    # lengths - 1 是因为序列索引是从0开始的
    last_step_indices = lengths.long() - 1

    # 创建一个 (batch_size, 1, embedding_dim) 的索引，用于 gather
    # 但更简单的方法是用高级索引
    try:
        last_step_embeddings = full_tensor[torch.arange(batch_size), last_step_indices]
        # 现在 last_step_embeddings 的形状是 [B, D]
        print(f"\n--- Analyzing LAST TIME STEP ONLY ---")
        debug_print(last_step_embeddings, name, step_counter)
        print("--- End Analysis ---")
    except IndexError as e:
        print(f"[DEBUG @ step {step_counter}] {name}: INDEXING ERROR!")
        print(f"  - Tensor shape: {full_tensor.shape}")
        print(f"  - Lengths min: {lengths.min()}, max: {lengths.max()}")
        print(f"  - Error: {e}")

def check_non_zero_counts(final_embeddings, past_lengths, step_counter=0):
    """
    检查并可视化一个批次中，每个序列的非零 hidden state 的位置。
    """
    print("\n" + "="*20 + f" Verifying Non-Zero Timestep Counts & Positions @ step {step_counter} " + "="*20)

    if final_embeddings.dim() != 3:
        print("Error: final_embeddings must be a 3D tensor [B, N, D]")
        return

    is_non_zero_mask = torch.any(final_embeddings != 0, dim=2)
    non_zero_counts = torch.sum(is_non_zero_mask, dim=1)
    lengths_match = torch.eq(non_zero_counts, past_lengths)

    print(f"Shape of final_embeddings: {final_embeddings.shape}")
    print(f"Expected lengths (past_lengths): {past_lengths.cpu().numpy()}")
    print(f"Actual non-zero counts:        {non_zero_counts.cpu().numpy()}")

    if torch.all(lengths_match):
        print("\n✅ SUCCESS: The number of non-zero hidden states perfectly matches past_lengths.")
    else:
        print("\n❌ MISMATCH DETECTED: The number of non-zero hidden states DOES NOT match past_lengths.")
        print(f"   - Number of matching sequences: {torch.sum(lengths_match)} / {final_embeddings.shape[0]}")

    # --- 新增的可视化部分 ---
    print("\n--- Visualizing Position of Non-Zero States for the FIRST sample ---")
    first_sample_len = int(past_lengths[0].item())
    first_sample_mask = is_non_zero_mask[0].int().cpu().numpy()

    print(f"First sample's actual length: {first_sample_len}")
    print(f"Non-zero mask (1 = active, 0 = zero):")
    print(first_sample_mask)

    # 构造预期的左Padding和右Padding模式
    total_len = len(first_sample_mask)
    expected_left_padding = [0]*(total_len - first_sample_len) + [1]*first_sample_len
    expected_right_padding = [1]*first_sample_len + [0]*(total_len - first_sample_len)

    print("\nComparing with expected patterns:")
    print(f"  - Actual Mask:             {list(first_sample_mask)}")
    print(f"  - Expected for LEFT Padding: {expected_left_padding}")
    print(f"  - Expected for RIGHT Padding:{expected_right_padding}")

    if list(first_sample_mask) == expected_right_padding:
        print("\n❗️❗️❗️ MAJOR FINDING: The output format is RIGHT-PADDED, but your input is LEFT-PADDED.")
    elif list(first_sample_mask) == expected_left_padding:
        print("\n✅ The output format is LEFT-PADDED, consistent with your input.")
    else:
        print("\n❗️ An UNEXPECTED pattern was found. The non-zero states are scattered.")

    print("="*80 + "\n")
class HSTU(GeneralizedInteractionModule):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,
    https://arxiv.org/abs/2402.17152.

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)
        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len
                            + max_output_len,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length,
                        self._max_sequence_length,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(
                        f"Initialize {name} as xavier normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str
    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        [B, N] -> [B, N, D].
        """


        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        float_dtype = user_embeddings.dtype
        causal_mask = torch.tril(torch.ones((N, N), device=device, dtype=torch.bool))
        padding_mask = torch.arange(N, device=device) >= (N - past_lengths.unsqueeze(1))
        combined_padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        attention_mask = causal_mask.unsqueeze(0) * combined_padding_mask
        final_attention_mask = attention_mask.to(user_embeddings.dtype)
        invalid_attention_mask = 1.0 - final_attention_mask 
        # 步骤2：检查核心 HSTU 模块
        user_embeddings, cached_states = self._hstu(
            x=user_embeddings,
            x_offsets=asynchronous_complete_cumsum_py(past_lengths),
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            invalid_attn_mask=final_attention_mask,
            # invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )

        final_embeddings = self._output_postproc(user_embeddings)

        return final_embeddings, cached_states
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Runs the main encoder.

        Args:
            past_lengths: (B,) x int64
            past_ids: (B, N,) x int64 where the latest engaged ids come first. In
                particular, past_ids[i, past_lengths[i] - 1] should correspond to
                the latest engaged values.
            past_embeddings: (B, N, D) x float or (\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            encoded_embeddings of [B, N, D].
        """
        #print(f"past_ids: {past_ids[0]},past_lengths: {past_lengths[0]},payloads:{past_payloads['timestamps'][0]}")
        encoded_embeddings, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache: Optional[List[HSTUCacheState]],
        return_cache_states: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Args:
            past_lengths: (B,) x int64.
            past_ids: (B, N,) x int64.
            past_embeddings: (B, N, D,) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            return_cache_states: bool.

        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )  # [B, N, D]
        current_embeddings = get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Runs encoder to obtain the current hidden states.

        Args:
            past_lengths: (B,) x int.
            past_ids: (B, N,) x int.
            past_embeddings: (B, N, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            (B, D,) x float, representing encoded states at the most recent time step.
        """
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )