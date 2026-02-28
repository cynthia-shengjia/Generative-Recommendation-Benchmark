from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
from typing import Dict, List, Any, Optional
from genrec.quantization.tokenizers.base_tokenizer import AbstractTokenizer
import logging
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BaseSeqRecDataset(Dataset):
    
    def __init__(
        self,
        data_interaction_files: str,
        data_text_files: str,
        tokenizer: AbstractTokenizer,
        config: dict,
        mode: str = 'train',  # 'train', 'valid', or 'test'
        device: Optional[str] = None
    ) -> None:
        self.config = config
        self.data_interaction_files = data_interaction_files
        self.data_text_files = data_text_files
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device
        
        self.vocab_size = self.tokenizer.vocab_size
        self.num_user_tokens = self.tokenizer.num_user_tokens
        self.n_codebooks = self.tokenizer.n_codebooks
        self.codebook_size = self.tokenizer.codebook_size
        self.digits = self.tokenizer.digits
        self.len_reserve_tokens = self.tokenizer.reserve_tokens
        self.dulicate_num = self.tokenizer.user_token_start_idx - (
            self.n_codebooks * self.codebook_size + self.len_reserve_tokens
        )
        
        assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
        self.user_seqs = self._load_user_seqs()
        self.user_ids = list(self.user_seqs.keys())
        
        self.tokens_per_item = self._get_tokens_per_item()
        if config['use_user_tokens']:
            self.max_token_len = self.tokens_per_item * self.config['max_seq_len'] + 1
        else:
            self.max_token_len = self.tokens_per_item * self.config['max_seq_len']
        
        self.all_items = self._get_all_items()
        self.samples = self._create_samples()
    
    def _get_all_items(self) -> List[int]:
        all_items = set()
        for item_seq in self.user_seqs.values():
            all_items.update(item_seq)
        return sorted(list(all_items))
    
    def _load_user_seqs(self) -> Dict[int, List[int]]:
        user_seqs = defaultdict(list)
        with open(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)
        for _, row in user_seqs_dataframe.iterrows():
            user_id = int(row['UserID'])
            item_seq = list(row["ItemID"])
            user_seqs[user_id] = item_seq
        return user_seqs
    
    def _get_tokens_per_item(self) -> int:
        if not self.tokenizer.item2tokens:
            return 1
        first_item = next(iter(self.tokenizer.item2tokens.keys()))
        return len(self.tokenizer.item2tokens[first_item])
    
    def _create_samples(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _create_samples()")
    
    def _get_item_tokens(self, item_id: int) -> List[int]:
        if item_id in self.tokenizer.item2tokens:
            return self.tokenizer.item2tokens[item_id]
        else:
            return [0] * self.tokens_per_item
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement __getitem__()")