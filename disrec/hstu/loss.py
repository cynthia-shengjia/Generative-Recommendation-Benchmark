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

import abc
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn.functional as F
from disrec.hstu.utils import NDPModule
from torch.utils.checkpoint import checkpoint
from typing import Callable
from disrec.hstu.fbgemm_replacement import (
    asynchronous_complete_cumsum_py,
    dense_to_jagged_py
)
class NegativesSampler(torch.nn.Module):

    def __init__(self, l2_norm: bool, l2_norm_eps: float) -> None:
        super().__init__()

        self._l2_norm: bool = l2_norm
        self._l2_norm_eps: float = l2_norm_eps

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_l2_norm(x)

    def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._l2_norm:
            x = x / torch.clamp(
                torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
                min=self._l2_norm_eps,
            )
        return x

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass

    @abc.abstractmethod
    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        pass

class LocalNegativesSampler(NegativesSampler):
    def __init__(
        self,
        num_items: int, # num_items 应该是你的 vocab_size
        item_emb: torch.nn.Embedding,
        l2_norm: bool,
        l2_norm_eps: float,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._num_items: int = num_items # 直接使用 vocab_size
        self._item_emb: torch.nn.Embedding = item_emb
    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_shape = positive_ids.size() + (num_to_sample,)
        # 直接生成物品 ID，假设 ID 从 1 开始，0 是 padding
        # torch.randint 的 high 是不包含的，所以 high=self._num_items 意味着生成 [1, self._num_items - 1] 的整数
        sampled_ids = torch.randint(
            low=1, 
            high=self._num_items, # self._num_items 是 vocab_size
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        # 不再需要 gather 操作
        return sampled_ids, self.normalize_embeddings(self._item_emb(sampled_ids))
# class LocalNegativesSampler(NegativesSampler):
#     def __init__(
#         self,
#         num_items: int,
#         item_emb: torch.nn.Embedding,
#         all_item_ids: List[int],
#         l2_norm: bool,
#         l2_norm_eps: float,
#     ) -> None:
#         super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

#         self._num_items: int = len(all_item_ids)
#         self._item_emb: torch.nn.Embedding = item_emb
#         self.register_buffer("_all_item_ids", torch.tensor(all_item_ids))

#     def debug_str(self) -> str:
#         sampling_debug_str = (
#             f"local{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
#         )
#         return sampling_debug_str

#     def process_batch(
#         self,
#         ids: torch.Tensor,
#         presences: torch.Tensor,
#         embeddings: torch.Tensor,
#     ) -> None:
#         pass

#     def forward(
#         self,
#         positive_ids: torch.Tensor,
#         num_to_sample: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Returns:
#             A tuple of (sampled_ids, sampled_negative_embeddings).
#         """
#         # assert torch.max(torch.abs(self._item_emb(positive_ids) - positive_embeddings)) < 1e-4
#         output_shape = positive_ids.size() + (num_to_sample,)
#         sampled_offsets = torch.randint(
#             low=0,
#             high=self._num_items,
#             size=output_shape,
#             dtype=positive_ids.dtype,
#             device=positive_ids.device,
#         )
#         sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
#         return sampled_ids, self.normalize_embeddings(self._item_emb(sampled_ids))

class InBatchNegativesSampler(NegativesSampler):

    def __init__(
        self,
        l2_norm: bool,
        l2_norm_eps: float,
        dedup_embeddings: bool,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._dedup_embeddings: bool = dedup_embeddings

    def debug_str(self) -> str:
        sampling_debug_str = (
            f"in-batch{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )
        if self._dedup_embeddings:
            sampling_debug_str += "-dedup"
        return sampling_debug_str

    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        """
        Args:
           ids: (N') or (B, N) x int64
           presences: (N') or (B, N) x bool
           embeddings: (N', D) or (B, N, D) x float
        """
        assert ids.size() == presences.size()
        assert ids.size() == embeddings.size()[:-1]
        if self._dedup_embeddings:
            valid_ids = ids[presences]
            unique_ids, unique_ids_inverse_indices = torch.unique(
                input=valid_ids, sorted=False, return_inverse=True
            )
            device = unique_ids.device
            unique_embedding_offsets = torch.empty(
                (unique_ids.numel(),),
                dtype=torch.int64,
                device=device,
            )
            unique_embedding_offsets[unique_ids_inverse_indices] = torch.arange(
                valid_ids.numel(), dtype=torch.int64, device=device
            )
            unique_embeddings = embeddings[presences][unique_embedding_offsets, :]
            self._cached_embeddings = self._maybe_l2_norm(unique_embeddings)
            self._cached_ids = unique_ids
        else:
            self._cached_embeddings = self._maybe_l2_norm(embeddings[presences])
            self._cached_ids = ids[presences]

    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings,).
        """
        X = self._cached_ids.size(0)
        sampled_offsets = torch.randint(
            low=0,
            high=X,
            size=positive_ids.size() + (num_to_sample,),
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        return (
            self._cached_ids[sampled_offsets],
            self._cached_embeddings[sampled_offsets],
        )


class AutoregressiveLoss(torch.nn.Module):

    @abc.abstractmethod
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Variant of forward() when the tensors are already in jagged format.

        Args:
            output_embeddings: [N', D] x float, embeddings for the current
                input sequence.
            supervision_ids: [N'] x int64, (positive) supervision ids.
            supervision_embeddings: [N', D] x float.
            supervision_weights: Optional [N'] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass

    @abc.abstractmethod
    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
            lengths: [B] x int32 representing number of non-zero elements per row.
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass


class BCELoss(AutoregressiveLoss):
    def __init__(
        self,
        temperature: float,
        # model: NDPModule,
        interaction_fn: Callable,
    ) -> None:
        super().__init__()
        self._temperature: float = temperature
        # self._model = model
        self._interaction_fn = interaction_fn
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=1,
        )

        positive_logits = (
            self._interaction_fn(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
                target_embeddings=supervision_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N']

        sampled_negatives_logits = (
            self._interaction_fn(
                input_embeddings=output_embeddings,  # [N', D]
                target_ids=sampled_ids,  # [N', 1]
                target_embeddings=sampled_negative_embeddings,  # [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N']
        sampled_negatives_valid_mask = (
            supervision_ids != sampled_ids.squeeze(1)
        ).float()  # [N']
        loss_weights = supervision_weights * sampled_negatives_valid_mask
        weighted_losses = (
            (
                F.binary_cross_entropy_with_logits(
                    input=positive_logits,
                    target=torch.ones_like(positive_logits),
                    reduction="none",
                )
                + F.binary_cross_entropy_with_logits(
                    input=sampled_negatives_logits,
                    target=torch.zeros_like(sampled_negatives_logits),
                    reduction="none",
                )
            )
            * loss_weights
            * 0.5
        )
        return weighted_losses.sum() / loss_weights.sum()

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
          lengths: [B] x int32 representing number of non-zero elements per row.
          output_embeddings: [B, N, D] x float, embeddings for the current
              input sequence.
          supervision_ids: [B, N] x int64, (positive) supervision ids.
          supervision_embeddings: [B, N, D] x float.
          supervision_weights: Optional [B, N] x float. Optional weights for
              masking out invalid positions, or reweighting supervision labels.
          negatives_sampler: sampler used to obtain negative examples paired with
              positives.
        Returns:
          (1), loss for the current engaged sequence.
        """
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        jagged_id_offsets = asynchronous_complete_cumsum_py(lengths)
        jagged_supervision_ids = (
            dense_to_jagged_py(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        jagged_supervision_weights = dense_to_jagged_py(
            supervision_weights.unsqueeze(-1),
            [jagged_id_offsets],
        )[0].squeeze(1)
        return self.jagged_forward(
            output_embeddings=dense_to_jagged_py(
                output_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_ids=jagged_supervision_ids,
            supervision_embeddings=dense_to_jagged_py(
                supervision_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_weights=jagged_supervision_weights,
            negatives_sampler=negatives_sampler,
        )


class BCELossWithRatings(AutoregressiveLoss):
    def __init__(
        self,
        temperature: float,
        # model: NDPModule,
        interaction_fn: Callable,
    ) -> None:
        super().__init__()
        self._temperature: float = temperature
        # self._model = model
        self._interaction_fn = interaction_fn
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        supervision_ratings: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        target_logits = (
            self._interaction_fn(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
                target_embeddings=supervision_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N', 1]

        # loss_weights = (supervision_ids > 0).to(torch.float32)

        weighted_losses = (
            F.binary_cross_entropy_with_logits(
                input=target_logits,
                target=supervision_ratings.to(dtype=target_logits.dtype),
                reduction="none",
            )
        ) * supervision_weights
        return weighted_losses.sum() / supervision_weights.sum()

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        supervision_ratings: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
          lengths: [B] x int32 representing number of non-zero elements per row.
          output_embeddings: [B, N, D] x float, embeddings for the current
              input sequence.
          supervision_ids: [B, N] x int64, (positive) supervision ids.
          supervision_embeddings: [B, N, D] x float.
          supervision_weights: Optional [B, N] x float. Optional weights for
              masking out invalid positions, or reweighting supervision labels.
          negatives_sampler: sampler used to obtain negative examples paired with
              positives.
        Returns:
          (1), loss for the current engaged sequence.
        """
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        jagged_id_offsets = asynchronous_complete_cumsum_py(lengths)
        jagged_supervision_ids = (
            dense_to_jagged_py(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        jagged_supervision_weights = dense_to_jagged_py(
            supervision_weights.unsqueeze(-1),
            [jagged_id_offsets],
        )[0].squeeze(1)
        return self.jagged_forward(
            output_embeddings=dense_to_jagged_py(
                output_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_ids=jagged_supervision_ids,
            supervision_embeddings=dense_to_jagged_py(
                supervision_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_weights=jagged_supervision_weights,
            supervision_ratings=dense_to_jagged_py(
                supervision_ratings.unsqueeze(-1),
                [jagged_id_offsets],
            )[0].squeeze(1),
            negatives_sampler=negatives_sampler,
        )


class SampledSoftmaxLoss(AutoregressiveLoss):

    def __init__(
        self,
        num_to_sample: int,
        softmax_temperature: float,
        # model: torch.nn.Module,
        interaction_fn: Callable,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self._num_to_sample: int = num_to_sample
        self._softmax_temperature: float = softmax_temperature
        # self._model = model
        self._interaction_fn = interaction_fn
        self._activation_checkpoint: bool = activation_checkpoint

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=self._num_to_sample,
        )
        positive_embeddings = negatives_sampler.normalize_embeddings(
            supervision_embeddings
        )
        positive_logits = (
            self._interaction_fn(
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
                target_embeddings=positive_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
            )
            / self._softmax_temperature
        )  # [0]
        sampled_negatives_logits = self._interaction_fn(
            input_embeddings=output_embeddings,  # [N', D]
            target_ids=sampled_ids,  # [N', R]
            target_embeddings=sampled_negative_embeddings,  # [N', R, D]
        )  # [N', R]  # [0]
        sampled_negatives_logits = torch.where(
            supervision_ids.unsqueeze(1) == sampled_ids,  # [N', R]
            -5e4,
            sampled_negatives_logits / self._softmax_temperature,
        )
        jagged_loss = -F.log_softmax(
            torch.cat([positive_logits, sampled_negatives_logits], dim=1), dim=1
        )[:, 0]
        return (jagged_loss * supervision_weights).sum() / supervision_weights.sum()

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
            lengths: [B] x int32 representing number of non-zero elements per row.
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        jagged_id_offsets = asynchronous_complete_cumsum_py(lengths)
        jagged_supervision_ids = (
            dense_to_jagged_py(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )

        args = OrderedDict(
            [
                (
                    "output_embeddings",
                    dense_to_jagged_py(
                        output_embeddings,
                        [jagged_id_offsets],
                    )[0],
                ),
                ("supervision_ids", jagged_supervision_ids),
                (
                    "supervision_embeddings",
                    dense_to_jagged_py(
                        supervision_embeddings,
                        [jagged_id_offsets],
                    )[0],
                ),
                (
                    "supervision_weights",
                    dense_to_jagged_py(
                        supervision_weights.unsqueeze(-1),
                        [jagged_id_offsets],
                    )[0].squeeze(1),
                ),
                ("negatives_sampler", negatives_sampler),
            ]
        )
        if self._activation_checkpoint:
            return checkpoint(
                self.jagged_forward,
                *args.values(),
                use_reentrant=False,
            )
        else:
            return self.jagged_forward(
                output_embeddings=dense_to_jagged_py(
                    output_embeddings,
                    [jagged_id_offsets],
                )[0],
                supervision_ids=jagged_supervision_ids,
                supervision_embeddings=dense_to_jagged_py(
                    supervision_embeddings,
                    [jagged_id_offsets],
                )[0],
                supervision_weights=dense_to_jagged_py(
                    supervision_weights.unsqueeze(-1),
                    [jagged_id_offsets],
                )[0].squeeze(1),
                negatives_sampler=negatives_sampler,
            )
        
import torch.nn as nn
class FullSoftmaxLoss(AutoregressiveLoss):
    """
    计算标准的全量 Softmax 交叉熵损失。
    该类的接口与 SampledSoftmaxLoss 完全兼容，可以作为直接替代品。
    """
    def __init__(
        self,
        item_embeddings: nn.Embedding,
        softmax_temperature: float = 1.0,
        activation_checkpoint: bool = False,
    ) -> None:
        """
        初始化函数。

        Args:
            item_embeddings (nn.Embedding): 包含所有物品嵌入的 nn.Embedding 层。
            softmax_temperature (float): 用于缩放 logits 的温度系数。
            activation_checkpoint (bool): 是否使用激活检查点以节省内存。
        """
        super().__init__()
        self._item_embeddings = item_embeddings
        self._softmax_temperature: float = softmax_temperature
        self._activation_checkpoint: bool = activation_checkpoint

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_weights: torch.Tensor,
        # 以下参数是为了接口兼容而接收，但不会被使用
        supervision_embeddings: torch.Tensor = None,
        negatives_sampler = None,
    ) -> torch.Tensor:
        """
        在 "jagged" (展平的) 张量上执行核心的损失计算。
        """
        # 获取完整的物品嵌入矩阵
        # all_item_embs -> (V, D)，其中 V 是词汇表大小
        all_item_embs = self._item_embeddings.weight

        # 计算与所有物品的 logits
        # output_embeddings [N', D], all_item_embs.T [D, V] -> logits [N', V]
        logits = torch.matmul(output_embeddings, all_item_embs.transpose(0, 1))

        # 应用温度缩放
        if self._softmax_temperature != 1.0:
            logits = logits / self._softmax_temperature
        
        # 使用 cross_entropy 计算损失
        # reduction='none' 使其返回每个样本的损失，以便我们可以应用 supervision_weights
        loss_per_token = F.cross_entropy(
            input=logits, 
            target=supervision_ids, 
            reduction='none'
        )

        # 应用权重并计算最终的平均损失，与 SampledSoftmaxLoss 的加权方式完全一致
        # (loss * weights).sum() / weights.sum()
        weighted_loss = (loss_per_token * supervision_weights).sum()
        total_weight = supervision_weights.sum()

        # 避免除以零
        if total_weight == 0:
            return torch.tensor(0.0, device=output_embeddings.device)
            
        return weighted_loss / total_weight

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler, # 接收但未使用
    ) -> torch.Tensor:
        """
        前向传播的入口点，处理从 dense 到 jagged 的张量转换。
        此方法的签名和逻辑与 SampledSoftmaxLoss 完全相同。
        """
        # 将 dense (带 padding) 的张量转换为 jagged (无 padding 的一维) 张量
        jagged_id_offsets = asynchronous_complete_cumsum_py(lengths)
        
        # 准备传递给 jagged_forward 的参数
        # 注意：我们只转换需要的参数，减少不必要的计算
        args = OrderedDict(
            [
                (
                    "output_embeddings",
                    dense_to_jagged_py(output_embeddings, [jagged_id_offsets])[0],
                ),
                (
                    "supervision_ids",
                    dense_to_jagged_py(supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets])[0].squeeze(1).long(),
                ),
                (
                    "supervision_weights",
                    dense_to_jagged_py(supervision_weights.unsqueeze(-1), [jagged_id_offsets])[0].squeeze(1),
                ),
                # 传递 None 给未使用的参数
                ("supervision_embeddings", None),
                ("negatives_sampler", None),
            ]
        )

        if self._activation_checkpoint:
            return checkpoint(self.jagged_forward, *args.values(), use_reentrant=False)
        else:
            return self.jagged_forward(**args)