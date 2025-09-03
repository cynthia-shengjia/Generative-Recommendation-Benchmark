from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
import numpy as np
from typing import Callable, Optional, Dict, List, Any, Tuple, Union
from genrec.tokenizers.GRTokenizer import AbstractTokenizer

class TokenizerAmazonReviews2014Dataset(Dataset):
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
        self.device = device if device else torch.device('cpu')
        
        # 确保配置中有必要的参数
        assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
        # 设置填充方向，默认为右填充
        self.padding_side = config.get('padding_side', 'right')
        assert self.padding_side in ['left', 'right'], "padding_side must be either 'left' or 'right'"
        
        # 忽略标签索引（用于不需要计算损失的位置）
        self.ignored_label = config.get('ignored_label', -100)
        
        # 加载数据
        self.item_reviews = self._load_item_reviews()
        self.user_seqs = self._load_user_seqs()
        self.user_ids = list(self.user_seqs.keys())
        
        # 计算每个物品的token数量（假设所有物品相同）
        self.tokens_per_item = self._get_tokens_per_item()
        self.max_token_len = (self.tokens_per_item + 1) * self.config['max_seq_len']
        
        # 直接创建样本，而不是先预处理整个序列
        self.samples = self._create_samples()

    def _load_item_reviews(self) -> Dict[int, str]:
        item_reviews = defaultdict(str)
        
        with open(self.data_text_files, 'rb') as f:
            item_titles_dataframe = pickle.load(f)
        
        for _, row in item_titles_dataframe.iterrows():
            item_id = int(row['ItemID'])
            item_context_info = row['Title']
            item_reviews[item_id] = item_context_info
        
        return item_reviews

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
        """获取每个物品的token数量（假设所有物品相同）"""
        if not self.tokenizer.item2tokens:
            return 1  # 默认值
        
        # 获取第一个物品的token数量
        first_item = next(iter(self.tokenizer.item2tokens.keys()))
        return len(self.tokenizer.item2tokens[first_item])

    def _pad_sequence(self, sequence: List[int], max_length: int) -> Tuple[List[int], List[int]]:
        """根据配置的填充方向填充序列并返回attention mask"""
        if len(sequence) < max_length:
            padding_length = max_length - len(sequence)
            if self.padding_side == 'left':
                # 左填充：在序列左侧添加padding
                padded_seq = [self.tokenizer.pad_token] * padding_length + sequence
                attention_mask = [0] * padding_length + [1] * len(sequence)
            else:  # right padding
                # 右填充：在序列右侧添加padding
                padded_seq = sequence + [self.tokenizer.pad_token] * padding_length
                attention_mask = [1] * len(sequence) + [0] * padding_length
        else:
            # 截断序列
            if self.padding_side == 'left':
                # 左填充时，保留序列的右侧部分
                padded_seq = sequence[-max_length:]
                attention_mask = [1] * max_length
            else:  # right padding
                # 右填充时，保留序列的左侧部分
                padded_seq = sequence[:max_length]
                attention_mask = [1] * max_length
        
        return padded_seq, attention_mask

    def _item_seq_to_token_seq(self, item_seq: List[int]) -> List[int]:
        """将物品序列转换为token序列"""
        token_seq = [token for item in item_seq for token in self.tokenizer.item2tokens[item]]
        return token_seq

    def _tokenize_first_n_items(self, item_seq: List[int]) -> Tuple[List[int], List[int], List[int], int]:
        """处理前n个物品，计算所有位置的损失"""
        token_seq = self._item_seq_to_token_seq(item_seq)
        
        # 输入序列是除了最后一个token的所有token
        input_seq = token_seq[:-1]
        seq_lens = len(input_seq)
        
        # 标签序列是除了第一个token的所有token（即输入序列的下一个token）
        label_seq = token_seq[1:]
        
        # 填充序列
        padded_input, attention_mask = self._pad_sequence(input_seq, self.max_token_len)
        padded_labels, _ = self._pad_sequence(label_seq, self.max_token_len)
        
        return padded_input, attention_mask, padded_labels, seq_lens

    def _tokenize_later_items(self, item_seq: List[int], pad_labels: bool = True) -> Tuple[List[int], List[int], List[int], int]:
        """处理后续物品，只计算最后一个位置的损失"""
        token_seq = self._item_seq_to_token_seq(item_seq)
        
        # 输入序列是除了最后一个物品的所有token
        input_seq = token_seq[:-self.tokens_per_item]
        seq_lens = len(input_seq)
        
        # 标签序列：只有最后一个物品对应的位置有真实标签，其他位置使用忽略标签
        label_seq = [self.ignored_label] * seq_lens
        
        # 添加最后一个物品的token作为标签
        last_item_tokens = token_seq[-self.tokens_per_item:]
        label_seq.extend(last_item_tokens)
        
        # 填充输入序列
        padded_input, attention_mask = self._pad_sequence(input_seq, self.max_token_len)
        
        # 填充标签序列（如果需要）
        if pad_labels:
            padded_labels, _ = self._pad_sequence(label_seq, self.max_token_len)
        else:
            padded_labels = label_seq  # 不填充，直接使用
        
        return padded_input, attention_mask, padded_labels, seq_lens

    def _create_samples(self) -> List[Dict[str, Any]]:
        """创建样本"""
        samples = []
        max_item_seq_len = self.config['max_seq_len']
        
        for user_id, item_seq in self.user_seqs.items():
            if self.mode == 'train':
                n_return_examples = max(len(item_seq) - max_item_seq_len, 1)
                
                # 处理第一个窗口（前n个物品）
                if len(item_seq) <= max_item_seq_len + 1:
                    # 如果序列长度不超过最大长度+1，直接处理所有物品
                    input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                        item_seq[:min(len(item_seq), max_item_seq_len + 1)]
                    )
                    samples.append({
                        'user_id': user_id,
                        'input_ids': torch.tensor(input_ids, dtype=torch.long),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                        'labels': torch.tensor(labels, dtype=torch.long),
                        'seq_lens': seq_lens
                    })
                else:
                    # 处理第一个窗口
                    first_window = item_seq[:max_item_seq_len + 1]
                    input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(first_window)
                    samples.append({
                        'user_id': user_id,
                        'input_ids': torch.tensor(input_ids, dtype=torch.long),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                        'labels': torch.tensor(labels, dtype=torch.long),
                        'seq_lens': seq_lens
                    })
                    
                    # 处理后续窗口
                    for i in range(1, n_return_examples):
                        cur_item_seq = item_seq[i:i + max_item_seq_len + 1]
                        input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                        samples.append({
                            'user_id': user_id,
                            'input_ids': torch.tensor(input_ids, dtype=torch.long),
                            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                            'labels': torch.tensor(labels, dtype=torch.long),
                            'seq_lens': seq_lens
                        })
            else:
                # 验证或测试模式
                if self.mode == 'test':
                    # 测试集：使用最后max_seq_len+1个物品，预测最后一个物品
                    if len(item_seq) < max_item_seq_len + 1:
                        window_items = item_seq
                    else:
                        window_items = item_seq[-(max_item_seq_len + 1):]
                else:  # valid
                    # 验证集：使用倒数第二个到倒数第max_seq_len+2个物品，预测倒数第二个物品
                    if len(item_seq) < max_item_seq_len + 2:
                        window_items = item_seq[:len(item_seq)-1]
                    else:
                        window_items = item_seq[-(max_item_seq_len + 2):-1]
                
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                    window_items, pad_labels=False
                )
                samples.append({
                    'user_id': user_id,
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels': torch.tensor(labels[-self.tokens_per_item:], dtype=torch.long),  # 只保留最后一个物品的标签
                    'seq_lens': seq_lens
                })
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[int, torch.Tensor]]:
        return self.samples[index]