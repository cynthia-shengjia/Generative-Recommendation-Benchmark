import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from disrec.sasrec.sasrec import SASRecModel
@dataclass
class SASRecOutput(ModelOutput):
    """
    SASRec模型的输出类
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[torch.FloatTensor] = None
    cache_states: Optional[List] = None


class SASRecConfig(PretrainedConfig):
    model_type = "SASRec"

    def __init__(
        self,
        vocab_size=3000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.5,
        initializer_range=0.02,
        pad_token_id=0,
        norm_emb: bool = False,
        max_seq_len = 20,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.temperature = 0.07
        self.item_num = vocab_size - 1
        self.pad_idx = pad_token_id
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.norm_emb = norm_emb
        self.dropout_rate = hidden_dropout_prob 

class SASRec4HF(PreTrainedModel):
    config_class = SASRecConfig
    base_model_prefix = "SASRec"
    
    def __init__(self, config: SASRecConfig):
        super().__init__(config)
        self.config = config

        # 创建SASRec模型
        self.SASRec = SASRecModel(config)
        self.loss_fn = nn.CrossEntropyLoss()
        # 初始化权重
        self.post_init()
    def get_input_embeddings(self):
        return self.SASRec.item_emb
    def set_input_embeddings(self, value):
        self.SASRec.item_emb = value
    
    def get_output_embeddings(self):
        return self.SASRec.item_emb


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[SASRecOutput, Tuple]:
        """
        Args:
            input_ids: (batch_size, sequence_length) 输入序列
            attention_mask: (batch_size, sequence_length) 注意力掩码
            labels: (batch_size,) 或 (batch_size, sequence_length) 标签
            return_sequence_embeddings: 是否返回完整序列嵌入
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        sequence_hidden_states = self.SASRec(
            log_seqs=input_ids
        ) 
        item_embeddings = self.get_input_embeddings().weight
        if self.SASRec.config.norm_emb: 
            item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)
            sequence_hidden_states = F.normalize(sequence_hidden_states, p=2, dim=-1)
        loss = None
        if not self.training:
            last_item_indices = torch.full((batch_size,), seq_len - 1, device=device)

            last_hidden_state = sequence_hidden_states.gather(
                1, 
                last_item_indices.view(-1, 1, 1).expand(-1, -1, self.config.hidden_size)
            ).squeeze(1) # -> [B, D]
            # 计算 logits: [B, D] @ [D, V] -> [B, V]
            logits = torch.matmul(last_hidden_state, item_embeddings.t())
            last_labels = labels.gather(1, last_item_indices.view(-1, 1)).squeeze(-1)
            loss = self.compute_loss(logits, last_labels)
            if self.SASRec.config.norm_emb:
                logits = logits / self.config.temperature
        else:
            # [B, N, D] @ [D, V] -> [B, N, V]
            logits = torch.matmul(sequence_hidden_states, item_embeddings.t())
            if self.SASRec.config.norm_emb:
                logits = logits / self.config.temperature
            loss = self.compute_loss(logits, labels)

        return SASRecOutput(
            loss=loss,
            logits=logits,
            # hidden_states=sequence_hidden_states, # 可以选择返回 hidden_states
            cache_states=None
        )
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.loss_fn(
            logits.view(-1, self.config.item_num + 1), 
            labels.view(-1)
        )
        return loss