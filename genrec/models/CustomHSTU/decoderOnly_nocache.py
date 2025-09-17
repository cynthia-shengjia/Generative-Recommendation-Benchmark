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
from typing import Optional, Dict, Tuple, Union,Any
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
def debug_print(tensor, name="tensor", check_last_vec=False):
    if tensor is None:
        print(f"[FINAL EVIDENCE] {name}: None")
        return
    tensor_cpu = tensor.detach().clone().to('cpu', non_blocking=True)
    
    print(
        f"[FINAL EVIDENCE] {name} (WHOLE TENSOR):\n"
        f"  - Shape: {tensor.shape}\n"
        f"  - Stats: min={tensor_cpu.min():.6f}, max={tensor_cpu.max():.6f}, "
        f"mean={tensor_cpu.mean():.6f}, std={tensor_cpu.std():.6f}\n"
        f"  - Has NaN: {torch.isnan(tensor_cpu).any()}"
    )
    # 新增：单独检查最后一个时间步的向量
    if check_last_vec and tensor.dim() == 3:
        last_vec_cpu = tensor_cpu[:, -1, :]
        print(
            f"[FINAL EVIDENCE] {name} (LAST VECTOR ONLY):\n"
            f"  - Shape: {last_vec_cpu.shape}\n"
            f"  - Stats: min={last_vec_cpu.min():.6f}, max={last_vec_cpu.max():.6f}, "
            f"mean={last_vec_cpu.mean():.6f}, std={last_vec_cpu.std():.6f}"
        )

        
class HSTUDecoderOnlyModel(PreTrainedModel):
    config_class = HSTUDecoderOnlyConfig
    
    def __init__(self, config: HSTUDecoderOnlyConfig):
        super().__init__(config)
        
        # Initialize shared embedding module
        self.embedding_module = LocalEmbeddingModule(
            config.vocab_size-1, 
            config.embedding_dim,
            padding_index=0,
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
        self.ignore_index = -100 
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        # Language modeling head
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.embedding_module._item_emb.weight
        self._tied_weights_keys = ["embedding_module._item_emb.weight", "lm_head.weight"]
        
        # self.post_init()
    
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
        
        # 处理 payloads 的两种情况
        actual_past_key_values = None
        past_payloads = None
        
        # 情况1：后续生成步骤，payloads 在 past_key_values 中
        if past_key_values is not None:
            if isinstance(past_key_values, tuple) and len(past_key_values) == 2:
                # 解码：(actual_past_key_values, past_payloads)
                actual_past_key_values, past_payloads = past_key_values
            else:
                actual_past_key_values = past_key_values
    
        
        # 第一次生成步骤，payloads 直接从 kwargs 传入
        if past_payloads is None and "past_payloads" in kwargs:
            past_payloads = kwargs["past_payloads"]
        
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": actual_past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        
        if past_payloads is not None:
            model_inputs["past_payloads"] = past_payloads
        
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
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
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and past_payloads is not None:
            # 获取输入和 payloads 的 batch size
            input_batch_size = input_ids.shape[0]
            payload_batch_size = past_payloads['timestamps'].shape[0]

            # 如果输入的 batch size 是 payloads 的整数倍，说明 generate 进行了 beam expansion
            if input_batch_size > payload_batch_size and input_batch_size % payload_batch_size == 0:
                num_beams = input_batch_size // payload_batch_size
                expanded_payloads = {}
                for key, value in past_payloads.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        # 使用 repeat_interleave 来扩展 payloads
                        # eg: tensor([1, 2]) with num_beams=3 -> tensor([1, 1, 1, 2, 2, 2])
                        expanded_payloads[key] = value.repeat_interleave(num_beams, dim=0)
                    else:
                        expanded_payloads[key] = value
                past_payloads = expanded_payloads
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        past_lengths = attention_mask.sum(dim=1)
        past_payloads = past_key_values[1] if past_key_values is not None and isinstance(past_key_values, tuple) and len(past_key_values) == 2 else past_payloads
        # print("past_lengths:", past_lengths[0])
        # print("payloads:", past_payloads['timestamps'][0])
        past_embeddings = self.embedding_module.get_item_embeddings(input_ids)

        # if hasattr(self, 'debug_counter'):
        self.debug_counter = 0
        torch.set_printoptions(profile="full")
        hidden_states = self.hstu(
            past_lengths=past_lengths,
            past_ids=input_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        logits = self.lm_head(hidden_states)
        import torch.nn.functional as F
        # ==================== 最终诊断代码 + Top-K 概率 ====================
        # if not self.training:

        #     # --- Top-K 概率分析 ---
        #     last_token_logits = logits[:, -1, :]
        #     probabilities = F.softmax(last_token_logits, dim=-1)
            
        #     k = 5
        #     top_k_probs, top_k_indices = torch.topk(probabilities, k)
        #     top_k_logits, _ = torch.topk(last_token_logits, k)

        #     print("\n--- Top 5 Predictions for Next Token ---")
        #     # 假设 batch size 为 1 进行打印
        #     print(f"{'Token ID':<12} | {'Logit':<18} | {'Probability':<18} |")
        #     print("-" * 55)
        #     for i in range(k):
        #         token_id = top_k_indices[0, i].item()
        #         logit_val = top_k_logits[0, i].item()
        #         prob_val = top_k_probs[0, i].item()
        #         is_winner = "<-- WINNER" if i == 0 else ""
        #         is_pad = "(PAD)" if token_id == 0 else ""
        #         print(f"{token_id:<12} | {logit_val:<18.6f} | {prob_val:<18.6f} | {is_winner} {is_pad}")
        #     print("#"*83 + "\n")
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                labels[..., 1:].reshape(-1)
            )
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (hidden_states,)
            return ((loss,) + output) if loss is not None else output
        updated_payloads = None
        if not self.training and past_payloads is not None:
            updated_payloads = {}
            for key, value in past_payloads.items():
                if key == 'timestamps' and value is not None:
                    last_timestamp = value[:, -1:]
                    # print("last_timestamp:", last_timestamp[0])
                    updated_timestamps = torch.cat([value, last_timestamp], dim=1)
                    updated_payloads[key] = updated_timestamps
                else:
                    updated_payloads[key] = value
        

        if updated_payloads is not None:
            encoded_past_key_values = (past_key_values, updated_payloads)
        else:
            encoded_past_key_values = past_key_values
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=encoded_past_key_values,  # 包含了 payloads
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None, 
        )
