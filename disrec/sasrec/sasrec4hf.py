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
        num_neg_samples=4,
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
        self.loss_type = "BCE"


class SASRec4HF(PreTrainedModel):
    config_class = SASRecConfig
    base_model_prefix = "SASRec"
    
    def __init__(self, config: SASRecConfig):
        super().__init__(config)
        self.config = config
        self.SASRec = SASRecModel(config)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()
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
            #logits: [B, D] @ [D, V] -> [B, V]
            logits = torch.matmul(last_hidden_state, item_embeddings.t())
            last_labels = labels.gather(1, last_item_indices.view(-1, 1)).squeeze(-1)
            loss = self.ce_loss_fn(logits, last_labels)
            # loss = self.compute_loss(last_hidden_state, last_labels, item_embeddings)
            if self.SASRec.config.norm_emb:
                logits = logits / self.config.temperature
                
        else:
            if self.config.loss_type = "BCE":
                last_hidden_state = sequence_hidden_states[:, -1:, :] # [B, 1, D]
                last_labels = labels[:, -1:] # [B, 1]
                loss = self.compute_loss(last_hidden_state, last_labels, input_ids, item_embeddings)
                logits = None
            elif self.config.loss_type = "softmax":
                last_hidden_state = sequence_hidden_states[:, -1, :]
                logits = torch.matmul(last_hidden_state, item_embeddings.t())
                if self.SASRec.config.norm_emb:
                    logits = logits / self.config.temperature
                loss = self.ce_loss_fn(logits, labels[:, -1])

        return SASRecOutput(
            loss=loss,
            logits=logits,
            cache_states=None
        )
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden_states.device
        batch_size, seq_len, _ = hidden_states.shape
        num_neg = self.config.num_neg_samples
        v_size = self.config.item_num + 1 

        probs = torch.ones(batch_size, v_size, device=device)
        
        forbidden_items = torch.cat([input_ids, labels], dim=-1) 

        clean_forbidden_items = forbidden_items.clone()
        clean_forbidden_items[clean_forbidden_items < 0] = self.config.pad_idx
        
        if clean_forbidden_items.max() >= v_size:
            raise ValueError(f"ID {clean_forbidden_items.max()} is larger than vocab_size {v_size}")

        probs.scatter_(1, clean_forbidden_items, 0.0) 
        probs[:, self.config.pad_idx] = 0.0

        neg_samples = torch.multinomial(probs, seq_len * num_neg, replacement=True)
        neg_items = neg_samples.view(batch_size, seq_len, num_neg)

        clean_labels = labels.clone()
        clean_labels[clean_labels < 0] = self.config.pad_idx
        
        pos_item_embs = item_embeddings[clean_labels] 
        pos_logits = (hidden_states * pos_item_embs).sum(dim=-1) 

        neg_item_embs = item_embeddings[neg_items]
        neg_logits = (hidden_states.unsqueeze(2) * neg_item_embs).sum(dim=-1)

        if self.config.norm_emb:
            pos_logits = pos_logits / self.config.temperature
            neg_logits = neg_logits / self.config.temperature
        all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1) 
        all_labels = torch.cat([
            torch.ones_like(pos_logits).unsqueeze(-1),
            torch.zeros_like(neg_logits)
        ], dim=-1)

        valid_mask = (labels >= 0) & (labels != self.config.pad_idx)
        
        if valid_mask.any():
            loss = self.loss_fn(all_logits[valid_mask], all_labels[valid_mask])
        else:
            loss = torch.tensor(0.0, device=device)

        return loss