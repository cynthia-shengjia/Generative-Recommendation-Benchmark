import abc
import torch.nn.functional as F
import torch
import math
from typing import Dict, Tuple, Optional
def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x
def init_mlp_xavier_weights_zero_bias(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0.0)
class NDPModule(torch.nn.Module):

    def forward(  # pyre-ignore[3]
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_sideinfo: Optional[torch.Tensor],
        item_ids: torch.Tensor,
        precomputed_logits: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_embeddings: (B, input_embedding_dim) x float
            item_embeddings: (1/B, X, item_embedding_dim) x float
            item_sideinfo: (1/B, X, item_sideinfo_dim) x float

        Returns:
            Tuple of (B, X,) similarity values, keyed outputs
        """
        pass
class EmbeddingModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=pad_token_id
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
class OutputPostprocessorModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pass


class L2NormEmbeddingPostprocessor(OutputPostprocessorModule):

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._eps: float = eps

    def debug_str(self) -> str:
        return "l2"

    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        return output_embeddings / torch.clamp(
            torch.linalg.norm(output_embeddings, ord=None, dim=-1, keepdim=True),
            min=self._eps,
        )


class LayerNormEmbeddingPostprocessor(OutputPostprocessorModule):

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._eps: float = eps

    def debug_str(self) -> str:
        return "ln"

    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        return F.layer_norm(
            output_embeddings,
            normalized_shape=(self._embedding_dim,),
            eps=self._eps,
        )
class InputFeaturesPreprocessorModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):

    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask




class LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):

    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):

    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim + rating_embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"posir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()

        user_embeddings = torch.cat(
            [past_embeddings, self._rating_emb(past_payloads["ratings"].int())],
            dim=-1,
        ) * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):

    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        self._rating_embedding_dim: int = rating_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_payloads["ratings"].int()),
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (
            self.get_preprocessed_masks(
                past_lengths,
                past_ids,
                past_embeddings,
                past_payloads,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask
def get_current_embeddings(
    lengths: torch.Tensor,
    encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int
        seq_embeddings: (B, N, D,) x float

    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.size()
    flattened_offsets = (lengths - 1) + torch.arange(
        start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
    ) * N
    return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)
class InteractionModule(torch.nn.Module):

    @abc.abstractmethod
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_item_sideinfo(
        self,
        item_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        pass

    @abc.abstractmethod
    def interaction(
        self,
        input_embeddings: torch.Tensor,  # [B, D]
        target_ids: torch.Tensor,  # [1, X] or [B, X]
        target_embeddings: Optional[torch.Tensor] = None,  # [1, X, D'] or [B, X, D']
    ) -> torch.Tensor:
        pass


class GeneralizedInteractionModule(InteractionModule):
    def __init__(
        self,
        ndp_module: NDPModule,
    ) -> None:
        super().__init__()

        self._ndp_module: NDPModule = ndp_module

    @abc.abstractmethod
    def debug_str(
        self,
    ) -> str:
        pass

    def interaction(
        self,
        input_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch._assert(
            len(input_embeddings.size()) == 2, "len(input_embeddings.size()) must be 2"
        )
        torch._assert(len(target_ids.size()) == 2, "len(target_ids.size()) must be 2")
        if target_embeddings is None:
            target_embeddings = self.get_item_embeddings(target_ids)
        torch._assert(
            len(target_embeddings.size()) == 3,
            "len(target_embeddings.size()) must be 3",
        )

        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            return self._ndp_module(
                input_embeddings=input_embeddings,  # [B, self._input_embedding_dim]
                item_embeddings=target_embeddings,  # [1/B, X, self._item_embedding_dim]
                item_sideinfo=self.get_item_sideinfo(
                    item_ids=target_ids
                ),  # [1/B, X, self._item_sideinfo_dim]
                item_ids=target_ids,
                precomputed_logits=None,
            )