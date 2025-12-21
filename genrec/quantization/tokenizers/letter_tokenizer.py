import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .Quant.rqvae_letter import LETTERRQVAE
from collections import defaultdict 
from .rqvae_tokenizer import RQVAETokenizer
import logging
import hashlib
import pickle

class LETTERRQVAETokenizer(RQVAETokenizer):
    """
    Workflow:
    1. Initialize TigerTokenizer: This creates the underlying RQ-VAE model.
    2. Train the model: Use an TigerTrainer to train the TigerTokenizer instance.
    3. Finalize: After training, call `finalize_tokenization()` to encode all items
       and build the item-to-token mapping.
    4. Tokenize: Now the tokenizer is ready to be used with `_tokenize_item_seq`.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.rq_vae = LETTERRQVAE(
            in_dim=self.config['sent_emb_dim'],
            num_emb_list=[self.codebook_size] * self.n_codebooks,
            e_dim=self.config['rq_e_dim'],
            layers=self.config['rq_layers'],
            dropout_prob=self.config['dropout_prob'],
            loss_type=self.config['loss_type'],
            quant_loss_weight=self.config['quant_loss_weight'],
            # LETTER
            diversity_beta=self.config['diversity_beta'],
            kmeans_init=self.config['rq_kmeans_init'],
            kmeans_iters=self.config['kmeans_iters'],
            commitment_beta=self.config['commitment_beta'],
        )
    def forward(self, embeddings: torch.Tensor, labels: dict):
        reconstructed_embeddings, quant_loss, indices, dense_reconstructed_embeddings = self.rq_vae(embeddings, labels)
        return reconstructed_embeddings, indices, quant_loss, dense_reconstructed_embeddings
    def encode(self, embeddings: torch.Tensor, labels = None) -> torch.Tensor:
        """
        Encodes embeddings into discrete codebook indices.
        Args:
            embeddings (torch.Tensor): A batch of embeddings.
        Returns:
            torch.Tensor: The codebook indices.
        """
        return self.rq_vae.get_indices(embeddings,labels)