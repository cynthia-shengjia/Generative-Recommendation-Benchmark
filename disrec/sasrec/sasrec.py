import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRecModel(torch.nn.Module):
    def __init__(self, config):
        super(SASRecModel, self).__init__()
        self.config = config
        self.item_num = config.item_num
        self.pad_idx = config.pad_idx

        self.item_emb = torch.nn.Embedding(self.item_num + 1, self.config.hidden_size, padding_idx=self.pad_idx)
        self.pos_emb = torch.nn.Embedding(self.config.max_seq_len, self.config.hidden_size)
        self.emb_dropout = torch.nn.Dropout(p=self.config.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.config.hidden_size, eps=1e-8)
        for _ in range(config.num_hidden_layers):
            new_attn_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                config.hidden_size, config.num_attention_heads, config.hidden_dropout_prob
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config.hidden_size, config.hidden_dropout_prob)
            self.forward_layers.append(new_fwd_layer)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def log2feats(self, log_seqs, return_attention=False):
        seqs = self.item_emb(log_seqs)  # batch_size x max_len x embedding_dim
        seqs *= self.item_emb.embedding_dim**0.5
        device = log_seqs.device
        positions = torch.arange(log_seqs.shape[1], dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand_as(log_seqs)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        #single_seq = torch.arange(log_seqs.shape[1], dtype=torch.long, device="cuda")
        # positions = single_seq.unsqueeze(0).repeat(log_seqs.shape[0], 1)
        # seqs += self.pos_emb(positions)

        # seqs = self.emb_dropout(seqs) 
        timeline_mask = (log_seqs == self.config.pad_token_id)  # batch_size x max_len
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim ；True means not padding

        tl = seqs.shape[1]  # time dim len for enforce causality
        causal_mask  = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=device))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)

            mha_outputs, attn_output_weights = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=causal_mask
            )  # query key value

            # mha_outputs.shape 10 x 256 x 64
            # attn_output_weights.shape 256 x 10 x 10 

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        log_feats *= ~timeline_mask.unsqueeze(-1)

        if return_attention:
            return log_feats, attn_output_weights[:, -1, :]
        else:
            return log_feats

    def forward(self, log_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # batch_size x max_len x embedding_dim
        return log_feats
    