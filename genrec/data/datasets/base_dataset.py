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
    """序列推荐数据集基类"""
    
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
        
        # 词汇表属性
        self.vocab_size = self.tokenizer.vocab_size
        self.num_user_tokens = self.tokenizer.num_user_tokens
        self.n_codebooks = self.tokenizer.n_codebooks
        self.codebook_size = self.tokenizer.codebook_size
        self.digits = self.tokenizer.digits
        self.len_reserve_tokens = self.tokenizer.reserve_tokens
        self.dulicate_num = self.tokenizer.user_token_start_idx - (
            self.n_codebooks * self.codebook_size + self.len_reserve_tokens
        )
        
        # 确保配置中有必要的参数
        assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
        # 加载数据
        self.user_seqs = self._load_user_seqs()
        self.user_ids = list(self.user_seqs.keys())
        
        # 计算每个物品的token数量
        self.tokens_per_item = self._get_tokens_per_item()
        self.max_token_len = self.tokens_per_item * self.config['max_seq_len'] + 1
        
        # 创建样本
        self.all_items = self._get_all_items() # 会被重复运行很多很多次
        self.samples = self._create_samples()
    
    def _get_all_items(self) -> List[int]:
        """获取所有物品ID"""
        all_items = set()
        for item_seq in self.user_seqs.values():
            all_items.update(item_seq)
        return sorted(list(all_items))
    
    def _load_user_seqs(self) -> Dict[int, List[int]]:
        """加载用户交互序列"""
        user_seqs = defaultdict(list)
        with open(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)
        for _, row in user_seqs_dataframe.iterrows():
            user_id = int(row['UserID'])
            item_seq = list(row["ItemID"])
            user_seqs[user_id] = item_seq
        return user_seqs
    
    def _get_tokens_per_item(self) -> int:
        """获取每个物品的token数量"""
        if not self.tokenizer.item2tokens:
            return 1
        first_item = next(iter(self.tokenizer.item2tokens.keys()))
        return len(self.tokenizer.item2tokens[first_item])
    
    def _create_samples(self) -> List[Dict[str, Any]]:
        """创建样本 - 子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement _create_samples()")
    
    def _get_item_tokens(self, item_id: int) -> List[int]:
        """获取物品的token序列"""
        if item_id in self.tokenizer.item2tokens:
            return self.tokenizer.item2tokens[item_id]
        else:
            return [0] * self.tokens_per_item
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个样本 - 子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement __getitem__()")