from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
import numpy as np
from typing import Callable, Optional, Dict, List, Any, Tuple, Union
from genrec.tokenizers.GRTokenizer import AbstractTokenizer

class SeqModelTrainingDataset(Dataset):
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
        self.max_token_len = (self.tokens_per_item + 1) * self.config['max_seq_len'] + 1
        # [user_id]: 1, senmatic_tokens:  (self.tokens_per_item + 1) * self.config['max_seq_len'] 
        
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


    def _decoder_item_seq_to_token_seq(self, item_seq: List[int]) -> List[int]:
        """
            [将物品序列转换为token序列]
        """
        token_seq = [token for item in item_seq for token in self.tokenizer.item2tokens[item]]
        return token_seq

    def _encoder_item_seq_to_token_seq(self, user_id, item_seq: List[int]) -> List[int]:
        """
            [user_id 2 token] + [将物品序列转换为token序列]
        """
        token_seq = [self.tokenizer.get_user_token(user_id)] + [token for item in item_seq for token in self.tokenizer.item2tokens[item]]
        return token_seq

    def _prepare_t5_inputs(self, user_id, item_seq: List[int], is_target: bool = False) -> Tuple[List[int], List[int], List[int], int]:
        """为T5模型准备输入和标签，遵循teacher forcing模式"""
        
        if is_target:
            token_seq = self._decoder_item_seq_to_token_seq(item_seq)
            # 对于目标序列，我们需要添加EOS token
            token_seq = token_seq + [self.tokenizer.eos_token]
            
            # 对于decoder输入，我们需要添加起始token并移除最后一个token
            decoder_input_ids = [self.tokenizer.bos_token] + token_seq[:-1]
            
            # 标签就是完整的目标序列
            labels = token_seq
            
            # 填充decoder输入
            padded_decoder_input, decoder_attention_mask = self._pad_sequence(
                decoder_input_ids, self.max_token_len
            )
            
            # 填充标签
            padded_labels, _ = self._pad_sequence(labels, self.max_token_len)
            
            return padded_decoder_input, decoder_attention_mask, padded_labels, len(decoder_input_ids)
        else:
            token_seq = self._encoder_item_seq_to_token_seq(user_id,item_seq)
            # 对于编码器输入，直接使用序列
            encoder_input_ids = token_seq
            
            # 填充编码器输入
            padded_encoder_input, encoder_attention_mask = self._pad_sequence(
                encoder_input_ids, self.max_token_len
            )
            
            return padded_encoder_input, encoder_attention_mask, None, len(encoder_input_ids)

    def _create_samples(self) -> List[Dict[str, Any]]:
        """创建样本，适应T5的encoder-decoder架构"""
        samples = []
        max_item_seq_len = self.config['max_seq_len']
        
        for user_id, item_seq in self.user_seqs.items():
            if self.mode == 'train':
                n_return_examples = max(len(item_seq) - max_item_seq_len, 1)
                
                # 处理第一个窗口（前n个物品）
                if len(item_seq) <= max_item_seq_len + 1:
                    # 编码器输入：前n-1个物品
                    encoder_input_ids, encoder_attention_mask, _, seq_lens = self._prepare_t5_inputs(
                        user_id,
                        item_seq[:-1]
                    )
                    
                    decoder_input_ids, decoder_attention_mask, labels, _ = self._prepare_t5_inputs(
                        user_id,
                        [item_seq[-1]], is_target=True
                    )
                    
                    samples.append({
                        'user_id': user_id,
                        'encoder_input_ids': torch.tensor(encoder_input_ids, dtype=torch.long),
                        'encoder_attention_mask': torch.tensor(encoder_attention_mask, dtype=torch.long),
                        'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
                        'decoder_attention_mask': torch.tensor(decoder_attention_mask, dtype=torch.long),
                        'labels': torch.tensor(labels, dtype=torch.long),
                        'seq_lens': seq_lens
                    })
                else:
                    # 处理第一个窗口
                    # 编码器输入：前max_seq_len个物品
                    encoder_input_ids, encoder_attention_mask, _, seq_lens = self._prepare_t5_inputs(
                        user_id,
                        item_seq[:max_item_seq_len]
                    )
                    
                    # 解码器输入和标签：第max_seq_len+1个物品
                    decoder_input_ids, decoder_attention_mask, labels, _ = self._prepare_t5_inputs(
                        user_id,
                        [item_seq[max_item_seq_len]], is_target=True
                    )
                    
                    samples.append({
                        'user_id': user_id,
                        'encoder_input_ids': torch.tensor(encoder_input_ids, dtype=torch.long),
                        'encoder_attention_mask': torch.tensor(encoder_attention_mask, dtype=torch.long),
                        'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
                        'decoder_attention_mask': torch.tensor(decoder_attention_mask, dtype=torch.long),
                        'labels': torch.tensor(labels, dtype=torch.long),
                        'seq_lens': seq_lens
                    })
                    
                    # 处理后续窗口
                    for i in range(1, n_return_examples):
                        # 编码器输入：从i到i+max_seq_len-1的物品
                        encoder_input_ids, encoder_attention_mask, _, seq_lens = self._prepare_t5_inputs(
                            user_id,
                            item_seq[i:i+max_item_seq_len]
                        )
                        
                        # 解码器输入和标签：第i+max_seq_len个物品
                        decoder_input_ids, decoder_attention_mask, labels, _ = self._prepare_t5_inputs(
                            user_id,
                            [item_seq[i+max_item_seq_len]], is_target=True
                        )
                        
                        samples.append({
                            'user_id': user_id,
                            'encoder_input_ids': torch.tensor(encoder_input_ids, dtype=torch.long),
                            'encoder_attention_mask': torch.tensor(encoder_attention_mask, dtype=torch.long),
                            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
                            'decoder_attention_mask': torch.tensor(decoder_attention_mask, dtype=torch.long),
                            'labels': torch.tensor(labels, dtype=torch.long),
                            'seq_lens': seq_lens
                        })
            else:
                # 验证或测试模式
                if self.mode == 'test':
                    # 测试集：使用最后max_seq_len个物品作为编码器输入，预测最后一个物品
                    if len(item_seq) <= max_item_seq_len:
                        encoder_items = item_seq[:-1]  # 所有物品除了最后一个
                        target_item = [item_seq[-1]]   # 最后一个物品
                    else:
                        encoder_items = item_seq[-(max_item_seq_len+1):-1]  # 最后max_seq_len个物品（除了最后一个）
                        target_item = [item_seq[-1]]  # 最后一个物品
                else:  # valid
                    # 验证集：使用倒数第二个到倒数第max_seq_len+1个物品作为编码器输入，预测倒数第二个物品
                    if len(item_seq) <= max_item_seq_len + 1:
                        encoder_items = item_seq[:-2]  # 所有物品除了最后两个
                        target_item = [item_seq[-2]]   # 倒数第二个物品
                    else:
                        encoder_items = item_seq[-(max_item_seq_len+2):-2]  # 从倒数第max_seq_len+2到倒数第三个物品
                        target_item = [item_seq[-2]]  # 倒数第二个物品
                
                # 准备编码器输入
                encoder_input_ids, encoder_attention_mask, _, seq_lens = self._prepare_t5_inputs(user_id,encoder_items)
                
                # 准备解码器输入和标签
                decoder_input_ids, decoder_attention_mask, labels, _ = self._prepare_t5_inputs(
                    user_id,
                    target_item, is_target=True
                )
                
                samples.append({
                    'user_id': user_id,
                    'encoder_input_ids': torch.tensor(encoder_input_ids, dtype=torch.long),
                    'encoder_attention_mask': torch.tensor(encoder_attention_mask, dtype=torch.long),
                    'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
                    'decoder_attention_mask': torch.tensor(decoder_attention_mask, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'seq_lens': seq_lens
                })
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[int, torch.Tensor]]:
        return self.samples[index]