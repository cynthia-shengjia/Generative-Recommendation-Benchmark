# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152).

This version has been modified to an Encoder-Decoder architecture.
"""

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fbgemm_replacement import (
    asynchronous_complete_cumsum_py,
    jagged_to_padded_dense_py,
    dense_to_jagged_py,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
import einops
import logging
import fbgemm_gpu
from utils import (
    EmbeddingModule,
    InputFeaturesPreprocessorModule,
    OutputPostprocessorModule
)
# from generative_recommenders.modeling.sequential.utils import get_current_embeddings
# from generative_recommenders.modeling.similarity_module import (
#     GeneralizedInteractionModule,
# )

def cumulative_sum_with_zero_cat(lengths: torch.Tensor) -> torch.Tensor:
    """
    使用 torch.cat 实现同样的功能，代码更简洁。
    """
    # 创建一个零张量
    zero = torch.tensor([0], device=lengths.device, dtype=lengths.dtype)
    # 计算累积和
    cumsum = torch.cumsum(lengths, dim=0)
    # 拼接起来
    return torch.cat([zero, cumsum])

TIMESTAMPS_KEY = "timestamps"


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
        B, N = all_timestamps.shape
        # B = all_timestamps.size(0)
        # N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


#  (v, padded_q, padded_k, new_outputs)
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
        qk_attn = qk_attn + rel_attn_bias(all_timestamps).unsqueeze(1)
    qk_attn = F.silu(qk_attn) / n
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
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
        # Final layer norm
        self._final_layer_norm = nn.LayerNorm(modules[0]._embedding_dim, eps=modules[0]._eps)

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
        
        x = self._final_layer_norm(x)
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


class HSTU(nn.Module):
    """
    The Encoder part of the Encoder-Decoder model, based on the original HSTU.
    It now functions primarily as an encoder, taking source sequences and
    producing hidden states.
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
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
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
                            max_seq_len=max_sequence_len
                            + max_output_len,
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
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None, # Added for compatibility
    ) -> torch.Tensor:
        """
        Runs the main encoder.

        Args:
            past_lengths: (B,) x int64
            past_ids: (B, N,) x int64
            past_embeddings: (B, N, D) x float or (\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            attention_mask: Not used by HSTU encoder's self-attention, but kept for API consistency.

        Returns:
            encoded_embeddings of [B, N, D].
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
        
        # Note: HSTU uses its internal causal masking logic.
        # The standard 'attention_mask' from huggingface is not directly used here.
        # It handles padding via jagged tensors.
        float_dtype = user_embeddings.dtype
        encoded_embeddings, _ = self._hstu(
            x=user_embeddings,
            x_offsets=asynchronous_complete_cumsum_py(past_lengths),
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            # This mask is for causal attention, not padding.
            invalid_attn_mask=1.0 - self._attn_mask[:N, :N].to(float_dtype),
            delta_x_offsets=None,
            cache=None,
            return_cache_states=False,
        )
        return self._output_postproc(encoded_embeddings)


class T5Attention(nn.Module):
    """A standard multi-head attention module, similar to T5."""
    def __init__(self, embedding_dim, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.key_value_proj_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        
        self.q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.o = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, key_value_states=None, attention_mask=None):
        """
        `key_value_states` are the encoder_hidden_states for cross-attention.
        If `key_value_states` is None, this is self-attention.
        """
        batch_size, seq_length, _ = hidden_states.size()

        if key_value_states is None:
            key_value_states = hidden_states
        
        q = self.q(hidden_states)
        k = self.k(key_value_states)
        v = self.v(key_value_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.key_value_proj_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.key_value_proj_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.key_value_proj_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(3, 2))
        
        if attention_mask is not None:
            scores += attention_mask
            
        # Normalize scores
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Get context vector
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        
        return self.o(attn_output)

class T5FeedForward(nn.Module):
    """A standard feed-forward network, similar to T5."""
    def __init__(self, embedding_dim, ffn_dim, dropout_rate):
        super().__init__()
        self.wi = nn.Linear(embedding_dim, ffn_dim, bias=False)
        self.wo = nn.Linear(ffn_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = F.relu

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class HSTUDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attention = T5Attention(
            embedding_dim=config['embedding_dim'], 
            num_heads=config['num_heads'], 
            dropout_rate=config['attn_dropout_rate']
        )
        self.cross_attention = T5Attention(
            embedding_dim=config['embedding_dim'], 
            num_heads=config['num_heads'], 
            dropout_rate=config['attn_dropout_rate']
        )
        self.ffn = T5FeedForward(
            embedding_dim=config['embedding_dim'],
            ffn_dim=config['linear_dim'],
            dropout_rate=config['linear_dropout_rate']
        )
        
        self.self_attn_layer_norm = nn.LayerNorm(config['embedding_dim'], eps=1e-6)
        self.cross_attn_layer_norm = nn.LayerNorm(config['embedding_dim'], eps=1e-6)
        self.ffn_layer_norm = nn.LayerNorm(config['embedding_dim'], eps=1e-6)
        
        self.dropout = nn.Dropout(config['linear_dropout_rate'])

    def forward(self, hidden_states, encoder_hidden_states, self_attention_mask=None, cross_attention_mask=None):
        # Self-Attention
        normed_hidden_states = self.self_attn_layer_norm(hidden_states)
        attn_output = self.self_attention(
            hidden_states=normed_hidden_states,
            attention_mask=self_attention_mask
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        # Cross-Attention
        normed_hidden_states = self.cross_attn_layer_norm(hidden_states)
        attn_output = self.cross_attention(
            hidden_states=normed_hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=cross_attention_mask
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        # Feed-Forward
        normed_hidden_states = self.ffn_layer_norm(hidden_states)
        ffn_output = self.ffn(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(ffn_output)

        return hidden_states


class HSTUDecoder(nn.Module):
    def __init__(self, config, embedding_module):
        super().__init__()
        self.config = config
        self._embedding_module = embedding_module
        self.layers = nn.ModuleList([HSTUDecoderLayer(config) for _ in range(config['num_blocks'])])
        self.final_layer_norm = nn.LayerNorm(config['embedding_dim'], eps=1e-6)
        self.dropout = nn.Dropout(config['linear_dropout_rate'])
        
    def forward(self, decoder_input_ids, encoder_hidden_states, decoder_attention_mask=None, encoder_attention_mask=None):
        # 1. Get embeddings
        # Assuming embedding_module can handle decoder inputs
        input_embeddings = self._embedding_module.get_item_embeddings(decoder_input_ids)
        
        # 2. Positional embeddings if needed (T5 uses relative, but we can simplify here)
        # For simplicity, we are not adding separate positional embeddings for the decoder here.
        
        hidden_states = self.dropout(input_embeddings)

        # 3. Create masks
        if decoder_attention_mask is None:
            # Create a casual mask
            seq_len = decoder_input_ids.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=decoder_input_ids.device) * float('-inf'), diagonal=1)
            decoder_attention_mask = causal_mask
            
        # T5 uses a 4D mask, let's create it. [batch_size, num_heads, seq_len, seq_len]
        # For PyTorch attention, it should be [batch_size, 1, seq_len, seq_len] or just broadcasted.
        extended_decoder_mask = decoder_attention_mask[None, None, :, :]
        
        extended_encoder_mask = None
        if encoder_attention_mask is not None:
             extended_encoder_mask = (1.0 - encoder_attention_mask) * -10000.0
             extended_encoder_mask = extended_encoder_mask[:, None, None, :]


        # 4. Pass through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attention_mask=extended_decoder_mask,
                cross_attention_mask=extended_encoder_mask
            )
            
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states


class HSTUEncoderDecoderModel(nn.Module):
    """
    The complete Encoder-Decoder model.
    """
    def __init__(self, encoder: HSTU, decoder: HSTUDecoder, num_items: int, config): # 最好传入一个config对象
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.lm_head = nn.Linear(self.encoder._embedding_dim, num_items, bias=False)
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_payloads: Optional[Dict[str, torch.Tensor]] = None, # 保持你的特殊输入，但设为可选
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else True
        past_ids = input_ids
        
        # past_lengths 可以从 attention_mask 计算得到
        past_lengths = attention_mask.sum(dim=1)
        
        # past_embeddings 需要从 input_ids 通过 embedding 层获得
        past_embeddings = self.encoder._embedding_module.get_item_embeddings(past_ids)
        
        # 确保 past_payloads 是正确准备和传入的。这通常需要在自定义的 DataCollator 中完成。
        if past_payloads is None:
            # 若未传入时间戳，创建一个全零的时间戳张量
            past_payloads = {
                "timestamps": torch.zeros_like(input_ids, dtype=torch.long)
            }

        # 1. Encode the source sequence
        encoder_hidden_states = self.encoder(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            # encoder_attention_mask 传递给 decoder 的 cross-attention
        )

        # 2. Decode using encoder's output
        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask, # 使用原始的 attention_mask
            decoder_attention_mask=decoder_attention_mask
        )

        # 3. Get logits
        logits = self.lm_head(decoder_outputs)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # logits (batch_size, sequence_length, vocab_size)
            # labels (batch_size, sequence_length)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # ===================================================================
        # 3. 返回符合 Trainer 期望的输出
        # ===================================================================
        if not return_dict:
            output = (logits,) + (decoder_outputs,)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            # past_key_values=... # 如果支持，用于加速生成
            # decoder_hidden_states=...
            # encoder_last_hidden_state=encoder_hidden_states,
            # encoder_hidden_states=...
        )
    # def forward(
    #     self,
    #     past_lengths: torch.Tensor,
    #     past_ids: torch.Tensor,
    #     past_embeddings: torch.Tensor,
    #     past_payloads: Dict[str, torch.Tensor],
    #     decoder_input_ids: torch.Tensor,
    #     encoder_attention_mask: Optional[torch.Tensor] = None, # Mask for encoder padding
    #     decoder_attention_mask: Optional[torch.Tensor] = None, # Mask for decoder padding + causal
    # ):
    #     # 1. Encode the source sequence
    #     encoder_hidden_states = self.encoder(
    #         past_lengths=past_lengths,
    #         past_ids=past_ids,
    #         past_embeddings=past_embeddings,
    #         past_payloads=past_payloads
    #     )

    #     # 2. Decode using encoder's output
    #     decoder_outputs = self.decoder(
    #         decoder_input_ids=decoder_input_ids,
    #         encoder_hidden_states=encoder_hidden_states,
    #         encoder_attention_mask=encoder_attention_mask,
    #         decoder_attention_mask=decoder_attention_mask
    #     )

    #     # 3. Get logits
    #     logits = self.lm_head(decoder_outputs)

    #     return logits