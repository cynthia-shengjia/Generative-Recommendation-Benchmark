import os
import json
import numpy as np
import hashlib
import logging
from collections import defaultdict
from sklearn.cluster import KMeans

from .base_tokenizer import AbstractTokenizer

class RQKmeansTokenizer(AbstractTokenizer):
    def __init__(self, config: dict):
        AbstractTokenizer.__init__(self, config)

        self.item2tokens = {} 
        self.user2tokens = {} 
        self.tokens2item = {}
        self.reserve_tokens = 100
        self.pad_token = 0    
        self.eos_token = 1
        self.bos_token = self.pad_token
        self.ignored_label = -100
        
        self.n_codebooks = self.config['n_codebooks']
        self.codebook_size = self.config['codebook_size']
        self.digits = self.n_codebooks + 1 
        self.dulicate_num = 0
        
        self.num_user_tokens = 2000
        self.user_token_start_idx = None 
        
        self.codebooks = []
        
        self.save_path = self.config['save_path']
        self.user_save_path = self.config['save_path'].replace('.json', '_users.json')
        self.tokens2item_save_path = self.config['save_path'].replace('.json', '_tokens2item.json')

    def log(self, message):
        logging.info(message)
        print(message)

    def _encode_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        N, D = embeddings.shape
        residuals = embeddings.copy()
        tokens = np.zeros((N, self.n_codebooks), dtype=int)

        self.log(f'[TOKENIZER] Starting RQ-KMeans with {self.n_codebooks} levels, K={self.codebook_size}...')

        for level in range(self.n_codebooks):
            self.log(f'[TOKENIZER] Fitting K-Means for level {level + 1}/{self.n_codebooks}...')
            
            kmeans = KMeans(
                n_clusters=self.codebook_size,
                n_init='auto', 
                random_state=42
            )
            
            cluster_indices = kmeans.fit_predict(residuals)
            tokens[:, level] = cluster_indices
            
            centers = kmeans.cluster_centers_
            self.codebooks.append(centers)
            
            quantized_vectors = centers[cluster_indices]
            residuals = residuals - quantized_vectors
            
            error = np.mean(np.linalg.norm(residuals, axis=1))
            self.log(f'[TOKENIZER] Level {level + 1} finished. Mean L2 Error: {error:.4f}')

        return tokens

    @classmethod
    def load(cls, config):
        tokenizer = cls(config)
        if os.path.exists(tokenizer.save_path):
            with open(tokenizer.save_path, 'r', encoding='utf-8') as f:
                loaded_item_data = json.load(f)
                tokenizer.item2tokens = {int(k): tuple(v) for k, v in loaded_item_data.items()}
                
        if os.path.exists(tokenizer.tokens2item_save_path):
            with open(tokenizer.tokens2item_save_path, 'r', encoding='utf-8') as f:
                loaded_tokens_data = json.load(f)
                tokenizer.tokens2item = {tuple(map(int, k.strip('()').split(','))): v 
                                         for k, v in loaded_tokens_data.items()}
                                         
        if os.path.exists(tokenizer.user_save_path):
            with open(tokenizer.user_save_path, 'r', encoding='utf-8') as f:
                loaded_user_data = json.load(f)
                tokenizer.user2tokens = {int(k): v for k, v in loaded_user_data.items()}
                if tokenizer.user2tokens:
                    tokenizer.user_token_start_idx = min(tokenizer.user2tokens.values())

        print(f"Tokenizer loaded successfully from JSON files based on config path: {tokenizer.save_path}")
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return self.user_token_start_idx + self.num_user_tokens

    def _hash_user_id(self, user_id: str) -> int:
        hash_object = hashlib.md5(str(user_id).encode())
        hash_int = int(hash_object.hexdigest(), 16)
        user_token_offset = hash_int % self.num_user_tokens
        return self.user_token_start_idx + user_token_offset

    def _adjust_semantic_ids_for_duplicates(self, item2sem_ids: dict) -> dict:
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

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        item2tokens = {}
        for item, sem_ids in item2sem_ids.items():
            tokens = list(sem_ids)
            for i in range(self.digits):
                offset = self.reserve_tokens + (self.codebook_size * i)
                tokens[i] += offset
            item2tokens[item] = tuple(tokens)
        return item2tokens

    def finalize_tokenization(self, item_embeddings_data: tuple, user_ids: list = None):
        if os.path.exists(self.save_path):
            self.log(f'[TOKENIZER] Loading from {self.save_path}')
            with open(self.save_path, 'r', encoding='utf-8') as f:
                loaded_item_data = json.load(f)
                self.item2tokens = {int(k): v for k, v in loaded_item_data.items()}
            if os.path.exists(self.tokens2item_save_path):
                self.log(f'[TOKENIZER] Loading from {self.tokens2item_save_path}')
                with open(self.tokens2item_save_path, 'r', encoding='utf-8') as f:
                    loaded_tokens_data = json.load(f)
                    self.tokens2item = {tuple(map(int, k.strip('()').split(','))): v 
                                        for k, v in loaded_tokens_data.items()}
            with open(self.user_save_path, 'r', encoding='utf-8') as f:
                loaded_user_data = json.load(f)
                self.user2tokens = {int(k): v for k, v in loaded_user_data.items()}
                self.log(f'[TOKENIZER] Loaded {len(self.item2tokens)} items and {len(self.user2tokens)} users.')
                min_user_token = min(self.user2tokens.values())
                self.user_token_start_idx = min_user_token
            return
            
        self.log(f'[TOKENIZER] Cache file not found. Starting full tokenization process...')
        item_ids, sent_embs = item_embeddings_data
        
        if len(item_ids) != sent_embs.shape[0]:
            raise ValueError("Number of item IDs does not match the number of embeddings.")

        all_codes = self._encode_kmeans(sent_embs)

        item2sem_ids = {}
        for i, item_id in enumerate(item_ids):
            item2sem_ids[item_id] = tuple(all_codes[i].tolist())

        adjusted_item2sem_ids = self._adjust_semantic_ids_for_duplicates(item2sem_ids)
        self.item2tokens = self._sem_ids_to_tokens(adjusted_item2sem_ids)
        self.tokens2item = {v: k for k, v in self.item2tokens.items()}
        self.log(f'[TOKENIZER] Processing complete. Mapped {len(self.item2tokens)} items.')
        
        if user_ids:
            unique_user_ids = list(set(user_ids))
            self.log(f'[TOKENIZER] Processing {len(unique_user_ids)} unique users...')
            for user_id in unique_user_ids:
                self.user2tokens[user_id] = self._hash_user_id(user_id)
            self.log(f'[TOKENIZER] Processing complete. Mapped {len(self.user2tokens)} users.')
            
        is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"
        if is_main_process:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(self.item2tokens, f, ensure_ascii=False, indent=4)
            with open(self.tokens2item_save_path, 'w', encoding='utf-8') as f:
                str_keys_tokens2item = {str(k): v for k, v in self.tokens2item.items()}
                json.dump(str_keys_tokens2item, f, ensure_ascii=False, indent=4)
            with open(self.user_save_path, 'w', encoding='utf-8') as f:
                json.dump(self.user2tokens, f, ensure_ascii=False, indent=4)

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

    def tokens_to_item(self, tokens: tuple) -> int:
        if not self.tokens2item:
            raise RuntimeError("Tokenizer has not been finalized. Please run `finalize_tokenization`.")
        return self.tokens2item.get(tokens)

    def get_user_token(self, user_id: str) -> int:
        return self.user2tokens[user_id]