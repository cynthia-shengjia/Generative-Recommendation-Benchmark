# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Decoder-Only version of HSTU model for causal language modeling.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Tuple, Union
from .hstu_utils4decoderOnly import (
    EmbeddingModule, 
    LocalEmbeddingModule,
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    LayerNormEmbeddingPostprocessor,
    L2NormEmbeddingPostprocessor,
    DotProductSimilarity
)
from .HSTU import HSTU
# Import all the HSTU components from the original code
from .fbgemm_replacement import asynchronous_complete_cumsum_py

# Assuming the HSTU class and related components are available from the original code
TIMESTAMPS_KEY = "timestamps"


class HSTUDecoderOnlyConfig(PretrainedConfig):
    model_type = "hstu_decoder_only"

    def __init__(
        self,
        vocab_size=36,
        max_sequence_len=10,
        max_output_len=5,
        embedding_dim=32,
        num_blocks=2,
        num_heads=4,
        linear_dim=64,
        attention_dim=8,
        attn_dropout_rate=0.1,
        linear_dropout_rate=0.1,
        user_embedding_norm='l2_norm',
        normalization='rel_bias',
        linear_config='uvqk',
        linear_activation='silu',
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id, 
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.linear_dim = linear_dim
        self.attention_dim = attention_dim
        self.attn_dropout_rate = attn_dropout_rate
        self.linear_dropout_rate = linear_dropout_rate
        self.user_embedding_norm = user_embedding_norm
        self.normalization = normalization
        self.linear_config = linear_config
        self.linear_activation = linear_activation
        self.use_cache = use_cache
        self.is_decoder = True
        self.max_output_len = max_output_len

class HSTUDecoderOnlyModel(PreTrainedModel):
    config_class = HSTUDecoderOnlyConfig
    
    def __init__(self, config: HSTUDecoderOnlyConfig):
        super().__init__(config)
        
        # Initialize shared embedding module
        self.embedding_module = LocalEmbeddingModule(
            config.vocab_size-1, 
            config.embedding_dim
        )
        self.similarity_module = DotProductSimilarity()
        # Initialize input preprocessor
        # self.input_preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        #     max_sequence_len=config.max_sequence_len + config.max_output_len,
        #     embedding_dim=config.embedding_dim,
        #     dropout_rate=config.attn_dropout_rate,
        # )
        input_preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=config.max_sequence_len + config.max_output_len,
            embedding_dim=config.embedding_dim,
            dropout_rate=config.attn_dropout_rate,
        )
        # Initialize output postprocessor
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
            max_sequence_len=config.max_sequence_len,
            max_output_len = config.max_output_len,
            embedding_dim=config.embedding_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            linear_dim=config.linear_dim,
            attention_dim=config.attention_dim,
            embedding_module=self.embedding_module,
            similarity_module = self.similarity_module,
           input_features_preproc_module=input_preprocessor,
            output_postproc_module=self.output_postprocessor,
            normalization=config.normalization,
            linear_config=config.linear_config,
            linear_activation=config.linear_activation,
            linear_dropout_rate=config.linear_dropout_rate,
            attn_dropout_rate=config.attn_dropout_rate,
            enable_relative_attention_bias=True,
            concat_ua=False,
            verbose=True,
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        # Language modeling head
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.embedding_module._item_emb.weight
        self._tied_weights_keys = ["embedding_module._item_emb.weight", "lm_head.weight"]
        
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Module:
        """Returns the model's input embedding layer."""
        return self.embedding_module._item_emb

    def get_output_embeddings(self) -> nn.Module:
        """Returns the model's output embedding/logits layer."""
        return self.lm_head

    def set_input_embeddings(self, value):
        """Sets the model's input embedding layer."""
        self.embedding_module._item_emb = value

    def set_output_embeddings(self, new_embeddings):
        """Sets the model's output embedding layer."""
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        if "timestamps" in kwargs:
            model_inputs["timestamps"] = kwargs["timestamps"]
        if "past_payloads" in kwargs:
             model_inputs["past_payloads"] = kwargs["past_payloads"]
        
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search - HSTU doesn't use traditional cache."""
        return past_key_values

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        past_payloads: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the HSTU decoder-only model.
        
        Args:
            input_ids: Token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            past_payloads: Optional dict containing timestamps and other payload data
            labels: Labels for loss computation
            past_key_values: Not used in HSTU but kept for compatibility
            use_cache: Whether to use cache (not implemented in HSTU)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput object
            
        Returns:
            CausalLMOutputWithPast containing loss, logits, and other outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        batch_size, seq_len = input_ids.shape
        
        # Calculate sequence lengths for each batch item
        #past_lengths应该是batch_size大小，存储每个样本的真实长度
        #我们应该+1以包括BOS
        past_lengths = attention_mask.sum(dim=1)
        
        # Get embeddings
        past_embeddings = self.embedding_module.get_item_embeddings(input_ids)
        
        
        # Forward through HSTU
        hidden_states = self.hstu(
            past_lengths=past_lengths,
            past_ids=input_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        
        # Get logits from language modeling head
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_fct(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                labels[..., 1:].reshape(-1)
            )
        else:
            loss = None

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None, 
        )
