import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from disrec.hstu.hstu import HSTU
from disrec.hstu.loss import SampledSoftmaxLoss, BCELoss,FullSoftmaxLoss
from disrec.hstu.loss import InBatchNegativesSampler, LocalNegativesSampler
from disrec.hstu.utils import (
    LocalEmbeddingModule,
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    LayerNormEmbeddingPostprocessor,
    L2NormEmbeddingPostprocessor,
)
from disrec.hstu.similarity.mol import create_mol_interaction_module
from disrec.hstu.similarity.dot_product import DotProductSimilarity
@dataclass
class HSTUOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[torch.FloatTensor] = None
    cache_states: Optional[List] = None


class HSTUConfig(PretrainedConfig):

    model_type = "hstu"
    
    def __init__(
        self,
        vocab_size: int = 10000,
        max_seq_len: int = 20,
        max_output_len: int = 2,
        embedding_dim: int = 64,
        num_blocks: int = 16,
        num_heads: int = 8,
        #dv
        linear_dim: int = 8,
        #dqk
        attention_dim: int = 8,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        linear_activation: str = "silu",
        linear_dropout_rate: float = 0.5,
        attn_dropout_rate: float = 0.5,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        loss_type: str = "FullSoftmaxLoss",
        similarity_module: str = "DotProduct", # DotProduct or MoL
        num_negative_samples: int = 100,
        pad_token_id: int = 0,
        all_item_ids: List[int] =[],
        sampling_strategy: str = "in-batch",
        temperature:float = 0.05,
        num_negatives:int = 512,
        loss_activation_checkpoint: bool = False,
        item_l2_norm:bool =True,
        l2_norm_eps: float = 1e-6,
        bf16_training: bool = False,
        user_embedding_norm:str = "l2_norm",
        tie_word_embeddings = True,
        **kwargs
    ):
        self.tie_word_embeddings = tie_word_embeddings,
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_output_len = max_output_len
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.linear_dim = linear_dim
        self.attention_dim = attention_dim
        self.normalization = normalization
        self.linear_config = linear_config
        self.linear_activation = linear_activation
        self.linear_dropout_rate = linear_dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.enable_relative_attention_bias = enable_relative_attention_bias
        self.concat_ua = concat_ua
        self.loss_type = loss_type
        self.num_negative_samples = num_negative_samples
        self.pad_token_id = pad_token_id
        self.all_item_ids = all_item_ids
        self.sampling_strategy=sampling_strategy
        self.temperature=temperature
        self.num_negatives=num_negatives
        self.loss_activation_checkpoint=loss_activation_checkpoint
        self.item_l2_norm=item_l2_norm
        self.l2_norm_eps=l2_norm_eps
        self.bf16_training=bf16_training
        self.similarity_module=similarity_module
        self.user_embedding_norm=user_embedding_norm
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class HSTU4HF(PreTrainedModel):
    config_class = HSTUConfig
    base_model_prefix = "hstu"
    
    def __init__(self, config: HSTUConfig):
        super().__init__(config)
        self.config = config
        
        self.embedding_module = LocalEmbeddingModule(
            num_items=config.vocab_size - 1,
            item_embedding_dim=config.embedding_dim,
            pad_token_id=config.pad_token_id
        )
        print(config.similarity_module)
        if config.similarity_module == "DotProduct":
            self.similarity_module = DotProductSimilarity()
        elif config.similarity_module == "MoL":
            self.similarity_module,_ = create_mol_interaction_module(
                query_embedding_dim=config.embedding_dim,
                item_embedding_dim=config.embedding_dim,
                bf16_training=config.bf16_training,
            )
        else:
            raise ValueError(f"Unknown similarity module: {config.similarity_module}")
        self.input_preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=config.max_seq_len + config.max_output_len,
            embedding_dim=config.embedding_dim,
            dropout_rate=config.attn_dropout_rate,
        )
        self.output_postprocessor = (
            L2NormEmbeddingPostprocessor(
                embedding_dim=config.embedding_dim,
                eps=1e-6,
            )
            if config.user_embedding_norm == "l2_norm"
            else LayerNormEmbeddingPostprocessor(
                embedding_dim=config.embedding_dim,
                eps=1e-6,
            )
        )
        self.hstu = HSTU(
            max_sequence_len=config.max_seq_len,
            max_output_len=config.max_output_len,
            embedding_dim=config.embedding_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            linear_dim=config.linear_dim,
            attention_dim=config.attention_dim,
            normalization=config.normalization,
            linear_config=config.linear_config,
            linear_activation=config.linear_activation,
            linear_dropout_rate=config.linear_dropout_rate,
            attn_dropout_rate=config.attn_dropout_rate,
            embedding_module=self.embedding_module,
            similarity_module=self.similarity_module,
            input_features_preproc_module=self.input_preprocessor,
            output_postproc_module=self.output_postprocessor,
            enable_relative_attention_bias=config.enable_relative_attention_bias,
            concat_ua=config.concat_ua,
            verbose=True
        )
        self.sampling_strategy = config.sampling_strategy
        if config.loss_type == "BCELoss":
            self.loss_fn = BCELoss(temperature=config.temperature, interaction_fn=self.hstu.interaction)
        elif config.loss_type == "SampledSoftmaxLoss":
            self.loss_fn = SampledSoftmaxLoss(
                num_to_sample=config.num_negatives,       
                softmax_temperature=config.temperature,      
                interaction_fn=self.hstu.interaction,                                  
                activation_checkpoint=config.loss_activation_checkpoint,
            )
        elif config.loss_type == "FullSoftmaxLoss":
            self.loss_fn = FullSoftmaxLoss(
                item_embeddings=self.embedding_module._item_emb,
                softmax_temperature=config.temperature,
                activation_checkpoint=config.loss_activation_checkpoint,
            )
        if config.sampling_strategy == "in-batch":
            self.negatives_sampler = InBatchNegativesSampler(
                l2_norm=config.item_l2_norm,
                l2_norm_eps=config.l2_norm_eps,
                dedup_embeddings=True,
            )
        elif config.sampling_strategy == "local":
            # all_ids = list(range(1, config.vocab_size))
            self.negatives_sampler = LocalNegativesSampler(
                num_items=config.vocab_size,
                item_emb=self.embedding_module._item_emb,
                # all_item_ids=all_ids,
                l2_norm=config.item_l2_norm,
                l2_norm_eps=config.l2_norm_eps,
            )
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)
        self.output_projection.weight = self.embedding_module._item_emb.weight
        self.post_init()
        self._tied_weights_keys = [
            "output_projection.weight",
            "embedding_module._item_emb.weight",
            "hstu._embedding_module._item_emb.weight",
            
            "input_preprocessor._pos_emb.weight",
            "hstu._input_features_preproc._pos_emb.weight" 
        ]
    def get_input_embeddings(self):
        return self.embedding_module._item_emb
    def set_input_embeddings(self, value):
        self.embedding_module._item_emb = value
    
    def get_output_embeddings(self):
        return self.output_projection

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        timestamps: Optional[torch.LongTensor] = None,
        past_lengths: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[HSTUOutput, Tuple]:

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        

        if past_lengths is not None:
            computed_lengths = past_lengths
        elif attention_mask is not None:
            computed_lengths = attention_mask.sum(dim=1)
        else:
            computed_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        

        input_embeddings = self.embedding_module.get_item_embeddings(input_ids)
        

        past_payloads = {}
        if timestamps is not None:
            past_payloads["timestamps"] = timestamps
        
        sequence_embeddings = self.hstu(
            past_lengths=computed_lengths,
            past_ids=input_ids,
            past_embeddings=input_embeddings,
            past_payloads=past_payloads,
        ) 
        supervision_ids = input_ids
        if self.sampling_strategy == "in-batch":
            # get_item_embeddings currently assume 1-d tensor.
            in_batch_ids = supervision_ids.view(-1)
            in_batch_embeddings = self.embedding_module.get_item_embeddings(in_batch_ids)
            self.negatives_sampler.process_batch(
                ids=in_batch_ids,
                presences=(in_batch_ids != 0),
                embeddings=in_batch_embeddings.detach(),
            )
        else:
            self.negatives_sampler._item_emb = self.embedding_module._item_emb

        ar_mask = supervision_ids[:, 1:] != 0
        logits = None
        if not self.training:
            last_item_indices = computed_lengths - 1 # [B]
            
            # last_item_indices.view(-1, 1, 1) -> [B, 1, 1]
            # .expand(-1, -1, self.config.embedding_dim) -> [B, 1, D]
            last_hidden_state = sequence_embeddings.gather(
                1, 
                last_item_indices.view(-1, 1, 1).expand(-1, -1, self.config.embedding_dim)
            ).squeeze(1) # [B, D]

            all_item_embeddings = self.get_input_embeddings().weight # 形状: [V, D]

            # [B, D] @ [D, V] -> [B, V]
            logits = torch.matmul(last_hidden_state, all_item_embeddings.t())
        loss = None
        if labels is not None:
            loss = self.compute_loss(
                lengths=computed_lengths,  # [B],
                output_embeddings=sequence_embeddings[:, :-1, :],  # [B, N-1, D]
                supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
                supervision_embeddings=input_embeddings[:, 1:, :],  # [B, N - 1, D]
                supervision_weights=ar_mask.float(),
                negatives_sampler=self.negatives_sampler,
            )
        return HSTUOutput(
            loss=loss,
            logits=logits,
            # hidden_states=sequence_embeddings.detach(),
            cache_states=None
        )
    
    def compute_loss(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler,
        **kwargs
    ) -> torch.Tensor:
        loss = self.loss_fn(
            lengths=lengths,
            output_embeddings=output_embeddings,
            supervision_ids=supervision_ids,
            supervision_embeddings=supervision_embeddings,
            supervision_weights=supervision_weights,
            negatives_sampler=negatives_sampler,
        )
        return loss
    