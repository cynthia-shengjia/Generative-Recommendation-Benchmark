import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .Quant.rqvae import RQVAE
from collections import defaultdict 
from .GRTokenizer import AbstractTokenizer
import logging
import hashlib
class TigerTokenizer(AbstractTokenizer, nn.Module):
    """
    Workflow:
    1. Initialize TigerTokenizer: This creates the underlying RQ-VAE model.
    2. Train the model: Use an TigerTrainer to train the TigerTokenizer instance.
    3. Finalize: After training, call `finalize_tokenization()` to encode all items
       and build the item-to-token mapping.
    4. Tokenize: Now the tokenizer is ready to be used with `_tokenize_item_seq`.
    """
    def __init__(self, config: dict):
        nn.Module.__init__(self)
        AbstractTokenizer.__init__(self, config)

        self.item2tokens = {} 
        self.user2tokens = {} 
        self.reserve_tokens = 100
        self.pad_token = 0    
        self.eos_token = 1
        self.ignored_label = -100
        self.n_codebooks = self.config['n_codebooks']
        # digits includes the codebooks plus an extra one for deduplication.
        self.digits = self.n_codebooks + 1
        self.dulicate_num = 0
        self.codebook_size = self.config['codebook_size']
        self.num_user_tokens = 2000
        self.user_token_start_idx = None 
        self.rq_vae = RQVAE(
            in_dim=self.config['sent_emb_dim'],
            num_emb_list=[self.codebook_size] * self.n_codebooks,
            e_dim=self.config['rq_e_dim'],
            layers=self.config['rq_layers'],
            dropout_prob=self.config['dropout_prob'],
            loss_type=self.config['loss_type'],
            quant_loss_weight=self.config['quant_loss_weight'],
            kmeans_init=self.config['rq_kmeans_init'],
            kmeans_iters=self.config['kmeans_iters'],
            commitment_beta=self.config['commitment_beta'],
        )
        self.save_path = self.config['save_path']
        self.user_save_path = self.config['save_path'].replace('.json', '_users.json')
    def forward(self, embeddings: torch.Tensor):
        reconstructed_embeddings, quant_loss, indices, _ = self.rq_vae(embeddings)
        return reconstructed_embeddings, indices, quant_loss
    def initialize_rqvae(self, embeddings: np.ndarray):
        self.log('[TOKENIZER] Initializing codebooks with K-Means...')
        embeddings_tensor = torch.from_numpy(embeddings).to(self.config['device'])
        self.rq_vae.vq_initialization(embeddings_tensor)

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encodes embeddings into discrete codebook indices.
        Args:
            embeddings (torch.Tensor): A batch of embeddings.
        Returns:
            torch.Tensor: The codebook indices.
        """
        return self.rq_vae.get_indices(embeddings)


    # @property
    # def max_token_seq_len(self) -> int:
    #     return self.config['max_item_seq_len'] * self.n_codebooks + 1

    @property
    def vocab_size(self) -> int:
        return self.codebook_size * self.codebook_size + self.reserve_tokens + self.dulicate_num

    def log(self, message):
        logging.info(message)
        print(message)
    def _hash_user_id(self, user_id: str) -> int:
        hash_object = hashlib.md5(str(user_id).encode())
        hash_int = int(hash_object.hexdigest(), 16)
        user_token_offset = hash_int % self.num_user_tokens
        return self.user_token_start_idx + user_token_offset
    def _adjust_semantic_ids_for_duplicates(self, item2sem_ids: dict) -> dict:
        """
        Ensures that each semantic ID sequence is unique by appending a counter.
        Returns:
            dict: 
                - adjusted_item2sem_ids (dict): The new item-to-semantic-ID mapping.
        """
        adjusted_item2sem_ids = {}
        sem_id_counts = defaultdict(int)

        sorted_item_ids = sorted(item2sem_ids.keys())

        for item_id in sorted_item_ids:
            sem_id_tuple = item2sem_ids[item_id]

            duplicate_counter = sem_id_counts[sem_id_tuple]

            adjusted_sem_id = sem_id_tuple + (duplicate_counter,)

            adjusted_item2sem_ids[item_id] = adjusted_sem_id

            sem_id_counts[sem_id_tuple] += 1

        max_count = max(sem_id_counts.values()) if sem_id_counts else 0
        self.dulicate_num = max_count
        base_vocab_size = self.n_codebooks * self.codebook_size + self.reserve_tokens + self.dulicate_num
        self.user_token_start_idx = base_vocab_size
        return adjusted_item2sem_ids
    def finalize_tokenization(self, item_embeddings_data: tuple, user_ids: list = None):
        if os.path.exists(self.save_path):
            self.log(f'[TOKENIZER] Loading from {self.save_path}')
            with open(self.save_path, 'r', encoding='utf-8') as f:
                loaded_item_data = json.load(f)
                self.item2tokens = {int(k): v for k, v in loaded_item_data.items()}

                self.log(f'[TOKENIZER] Loading from {self.user_save_path}')
            with open(self.user_save_path, 'r', encoding='utf-8') as f:
                loaded_user_data = json.load(f)
                self.user2tokens = {int(k): v for k, v in loaded_user_data.items()}
                self.log(f'[TOKENIZER] Loaded {len(self.item2tokens)} items and {len(self.user2tokens)} users.')
                self.user_token_start_idx = len(self.item2tokens) + self.reserve_tokens
            return
        self.log(f'[TOKENIZER] Cache file not found. Starting full tokenization process...')
        item_ids, sent_embs = item_embeddings_data
        
        if len(item_ids) != sent_embs.shape[0]:
            raise ValueError("Number of item IDs does not match the number of embeddings.")

        self.eval()
        with torch.no_grad():
            all_embs_tensor = torch.from_numpy(sent_embs).to(self.config['device'])
            all_codes = self.encode(all_embs_tensor).cpu().numpy()

        item2sem_ids = {}
        for i, item_id in enumerate(item_ids):
            item2sem_ids[item_id] = tuple(all_codes[i].tolist())

        adjusted_item2sem_ids = self._adjust_semantic_ids_for_duplicates(item2sem_ids)
        

        self.item2tokens = self._sem_ids_to_tokens(adjusted_item2sem_ids)
        
        self.log(f'[TOKENIZER] Processing complete. Mapped {len(self.item2tokens)} items.')
        if user_ids:
            unique_user_ids = list(set(user_ids))
            self.log(f'[TOKENIZER] Processing {len(unique_user_ids)} unique users...')
            for user_id in unique_user_ids:
                self.user2tokens[user_id] = self._hash_user_id(user_id)
            self.log(f'[TOKENIZER] Processing complete. Mapped {len(self.user2tokens)} users.')
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.item2tokens, f, ensure_ascii=False, indent=4)
        with open(self.user_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.user2tokens, f, ensure_ascii=False, indent=4)
    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        item2tokens = {}
        for item, sem_ids in item2sem_ids.items():
            tokens = list(sem_ids)
            for i in range(self.digits):
                offset = self.reserve_tokens + (self.codebook_size * i)
                tokens[i] += offset
            item2tokens[item] = tuple(tokens)
        return item2tokens
    def _tokenize_item_seq(self, item_seq: list, max_item_len: int, user_id: str = None) -> list:
        if not self.item2tokens:
            raise RuntimeError("Tokenizer has not been finalized. Please run `finalize_tokenization` after training.")
        max_tokens_len = max_item_len * self.digits
        token_seq = [token for item in item_seq for token in self.item2tokens[item]]

        if len(token_seq) > max_tokens_len:
            token_seq = token_seq[-max_tokens_len:]

        padding_len = max_tokens_len - len(token_seq)
        if padding_len > 0:
            padding_tokens = [self.pad_token] * padding_len
            token_seq = padding_tokens + token_seq
        if user_id is not None:
            user_token = self.get_user_token(user_id)
            token_seq = [user_token] + token_seq
        token_seq.append(self.eos_token)
        return token_seq
    
    def get_user_token(self, user_id: str) -> int:
        return self.user2tokens[user_id]