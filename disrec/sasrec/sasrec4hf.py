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
        num_hidden_layers=4,
        num_attention_heads=4,
        hidden_dropout_prob=0.2,
        initializer_range=0.02,
        pad_token_id=0,
        norm_emb: bool = False,
        max_seq_len=20,
        num_neg_samples=1,
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
        self.num_neg_samples = num_neg_samples 


class SASRec4HF(PreTrainedModel):
    config_class = SASRecConfig
    base_model_prefix = "SASRec"
    
    def __init__(self, config: SASRecConfig):
        super().__init__(config)
        self.config = config

        # 创建SASRec模型
        self.SASRec = SASRecModel(config)
        # 修改为二元交叉熵损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()
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
            ).squeeze(1)  # -> [B, D]
            # 计算 logits: [B, D] @ [D, V] -> [B, V]
            logits = torch.matmul(last_hidden_state, item_embeddings.t())
            last_labels = labels.gather(1, last_item_indices.view(-1, 1)).squeeze(-1)
            loss = self.ce_loss_fn(logits, last_labels)
            # loss = self.compute_loss(last_hidden_state, last_labels, item_embeddings)
            if self.SASRec.config.norm_emb:
                logits = logits / self.config.temperature
                
        else:
            # [B, N, D] @ [D, V] -> [B, N, V]
            # last_hidden_state = sequence_hidden_states[:, -1, :]
            # logits = torch.matmul(last_hidden_state, item_embeddings.t())
            # if self.SASRec.config.norm_emb:
            #     logits = logits / self.config.temperature
            # loss = self.ce_loss_fn(logits, labels[:, -1])

            # 只计算最后一位的BCE
            # last_hidden_state = sequence_hidden_states[:, -1:, :] # [B, 1, D]
            # last_labels = labels[:, -1:] # [B, 1]
            # loss = self.compute_loss(last_hidden_state, last_labels, item_embeddings)
            # logits = None
            # 全部计算的BCE
            loss = self.compute_loss(sequence_hidden_states, labels, item_embeddings)
            logits = None


            # logits = torch.matmul(sequence_hidden_states, item_embeddings.t())
            # if self.SASRec.config.norm_emb:
            #     logits = logits / self.config.temperature
            # loss = self.compute_loss(sequence_hidden_states, labels, item_embeddings)
        # last_hidden_state = sequence_hidden_states[:, -1, :]
        # logits = torch.matmul(last_hidden_state, item_embeddings.t())
        # if self.SASRec.config.norm_emb:
        #     logits = logits / self.config.temperature
        # if labels is not None:
        #     # 提取最后一个位置的 Label
        #     # labels: [B, N] -> [B]
        #     last_labels = labels[:, -1]
            
        #     # 直接使用 CrossEntropyLoss 计算
        #     loss = self.ce_loss_fn(logits, last_labels)
        return SASRecOutput(
            loss=loss,
            logits=logits,
            cache_states=None
        )
    # def compute_loss(
    #         self,
    #         hidden_states: torch.Tensor,  # [B, N, D]
    #         labels: torch.Tensor,  # [B, N]
    #         item_embeddings: torch.Tensor,  # [V, D]
    #     ) -> torch.Tensor:
            
    #         # 1. 计算全量 Logits
    #         # hidden_states: [B, N, D]
    #         # item_embeddings.t(): [D, V]
    #         # logits: [B, N, V]
    #         logits = torch.matmul(hidden_states, item_embeddings.t())

    #         # 2. 如果开启了 Embedding 归一化，需要除以温度系数
    #         if self.SASRec.config.norm_emb:
    #             logits = logits / self.config.temperature

    #         # 3. 调整形状以适应 CrossEntropyLoss
    #         # Logits: [B * N, V]
    #         # Labels: [B * N]
    #         logits = logits.view(-1, logits.size(-1))
    #         labels = labels.view(-1)

    #         # 4. 计算 Loss
    #         # 这里会自动处理 Softmax，且根据 __init__ 设置忽略 padding
    #         loss = self.ce_loss_fn(logits, labels)
            
    #         return loss   
    def compute_loss(
        self,
        hidden_states: torch.Tensor,  # [B, D] 或 [B, N, D]
        labels: torch.Tensor,  # [B] 或 [B, N]
        item_embeddings: torch.Tensor,  # [V, D]
    ) -> torch.Tensor:
        """
        使用二元交叉熵损失，对每个位置采样正样本和负样本
        
        Args:
            hidden_states: 隐藏状态 [B, D] 或 [B, N, D]
            labels: 正样本标签 [B] 或 [B, N]
            item_embeddings: item嵌入矩阵 [V, D]
        """
        device = hidden_states.device
        
        if hidden_states.dim() == 2:
            # [B, D] -> [B, 1, D]
            hidden_states = hidden_states.unsqueeze(1)
            labels = labels.unsqueeze(1)  # [B] -> [B, 1]
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_neg = self.config.num_neg_samples
        
        # labels: [B, N] -> [B, N, 1, D]
        pos_item_embs = item_embeddings[labels]  # [B, N, D]
        # [B, N, D] * [B, N, D] -> [B, N]
        pos_logits = (hidden_states * pos_item_embs).sum(dim=-1)  # [B, N]
        
        # 负采样
        neg_items = torch.randint(
            1, self.config.item_num + 1,  # 避免采样到padding (0)
            size=(batch_size, seq_len, num_neg),
            device=device
        )
        
        # 避免负样本与正样本重复
        # 如果采样到正样本，重新采样
        pos_labels_expanded = labels.unsqueeze(-1)  # [B, N, 1]
        mask = (neg_items == pos_labels_expanded)  # [B, N, num_neg]
        while mask.any():
            new_samples = torch.randint(
                1, self.config.item_num + 1,
                size=(batch_size, seq_len, num_neg),
                device=device
            )
            neg_items = torch.where(mask, new_samples, neg_items)
            mask = (neg_items == pos_labels_expanded)
        
        neg_item_embs = item_embeddings[neg_items]  # [B, N, num_neg, D]
        #[B, N, 1, D] * [B, N, num_neg, D] -> [B, N, num_neg]
        neg_logits = (hidden_states.unsqueeze(2) * neg_item_embs).sum(dim=-1)
        
        if self.SASRec.config.norm_emb:
            pos_logits = pos_logits / self.config.temperature
            neg_logits = neg_logits / self.config.temperature
        
        # pos_logits: [B, N] -> [B, N, 1]
        # neg_logits: [B, N, num_neg]
        all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)  # [B, N, 1+num_neg]
        
        pos_labels_binary = torch.ones_like(pos_logits).unsqueeze(-1)  # [B, N, 1]
        neg_labels_binary = torch.zeros_like(neg_logits)  # [B, N, num_neg]
        all_labels = torch.cat([pos_labels_binary, neg_labels_binary], dim=-1)  # [B, N, 1+num_neg]
        
        valid_mask = (labels != self.config.pad_idx)  # [B, N]
        
        if valid_mask.any():
            valid_logits = all_logits[valid_mask]  # [num_valid, 1+num_neg]
            valid_labels = all_labels[valid_mask]  # [num_valid, 1+num_neg]
            
            loss = self.loss_fn(valid_logits, valid_labels)
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss