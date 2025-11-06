from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
from typing import Callable, Optional, Dict, List, Any, Tuple, Union
from genrec.tokenizers.GRTokenizer import AbstractTokenizer
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
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
        self.device = device
        # --- 词汇表属性 ---
        self.vocab_size = self.tokenizer.vocab_size
        self.num_user_tokens = self.tokenizer.num_user_tokens
        self.n_codebooks = self.tokenizer.n_codebooks
        self.codebook_size = self.tokenizer.codebook_size
        self.digits = self.tokenizer.digits
        self.len_reserve_tokens = self.tokenizer.reserve_tokens
        self.dulicate_num = self.tokenizer.user_token_start_idx - (self.n_codebooks * self.codebook_size + self.len_reserve_tokens)
        #reserve_tokens, digit 1 digit 2 digit 3, dulicate_num , user tokens
        # 确保配置中有必要的参数
        assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
        self.user_seqs = self._load_user_seqs()
        self.user_ids = list(self.user_seqs.keys())
        
        # 计算每个物品的token数量（假设所有物品相同）
        self.tokens_per_item = self._get_tokens_per_item()
        self.max_token_len = self.tokens_per_item * self.config['max_seq_len'] + 1
        self._precompute_vocab_ranges_and_masks()
        # 直接创建样本，而不是先预处理整个序列
        self.samples = self._create_samples()
    def _precompute_vocab_ranges_and_masks(self):
            """
            根据词汇表结构，预先计算每个块的允许 token ID 列表。
            顺序: reserve_tokens, digit 1, digit 2, digit 3, dulicate_num, user tokens
            """
            
            # 1. 计算每个块的起始索引
            start_reserve = 0
            end_reserve = self.len_reserve_tokens
            start_digit_1 = end_reserve
            end_digit_1 = start_digit_1 + self.codebook_size
            start_digit_2 = end_digit_1
            end_digit_2 = start_digit_2 + self.codebook_size
            start_digit_3 = end_digit_2
            end_digit_3 = start_digit_3 + self.codebook_size
            start_dulicate = end_digit_3
            end_dulicate = start_dulicate + self.dulicate_num
            start_user = end_dulicate
            end_user = start_user + self.num_user_tokens
            
            if end_user != self.vocab_size:
                logger.warning(
                    f"词汇表范围计算不匹配！计算得到的 'end_user' ({end_user}) 与 "
                    f"self.vocab_size ({self.vocab_size}) 不符。"
                )
                end_user = self.vocab_size

            logger.info("--- 词汇表范围 (用于 Loss Mask) ---")
            logger.info(f"  [0] Reserve:  [{start_reserve}, {end_reserve})")
            logger.info(f"  [1] Digit 1:  [{start_digit_1}, {end_digit_1})")
            logger.info(f"  [2] Digit 2:  [{start_digit_2}, {end_digit_2})")
            logger.info(f"  [3] Digit 3:  [{start_digit_3}, {end_digit_3})")
            logger.info(f"  [4] Dulicate: [{start_dulicate}, {end_dulicate})")
            logger.info(f"  [5] User:     [{start_user}, {end_user})")
            logger.info(f"  Total Vocab Size: {self.vocab_size}")
            
            # 2. [!change] 存储掩码的“构建块”
            
            # "第一个 token" (位置 0) 的掩码
            self.allowed_user_tokens = list(range(start_user, end_user))
            
            # "后续4个一循环" (位置 1, 2, 3, 4; 5, 6, 7, 8; ...) 的掩码
            self.cycle_masks = [
                list(range(start_digit_1, end_digit_1)),  # 循环 1
                list(range(start_digit_2, end_digit_2)),  # 循环 2
                list(range(start_digit_3, end_digit_3)),  # 循环 3
                list(range(start_dulicate, end_dulicate)) # 循环 4
            ]
    def _load_user_seqs(self) -> Dict[int, List[int]]:
        # 保持不变
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
        first_item = next(iter(self.tokenizer.item2tokens.keys()))
        return len(self.tokenizer.item2tokens[first_item])

    def _create_samples(self) -> List[Dict[str, Any]]:
        """创建样本，返回历史物品序列和目标物品"""
        samples = []
        max_item_seq_len = self.config['max_seq_len']
        
        for user_id, item_seq in self.user_seqs.items():
            if self.mode == 'train':
                # 训练集需要截断倒数的两个item (倒数第二个item作为valid，倒数第一个item作为test)
                item_seq = item_seq[:-2]
                for i in range(1, len(item_seq)):
                    history = item_seq[:i]
                    target = item_seq[i]
                    if len(history) > max_item_seq_len:
                        history = history[-max_item_seq_len:]
                    samples.append({
                        'user_id': user_id,
                        'history_items': history,
                        'target_item': target
                    })
            elif self.mode == 'merge_train':
                # 训练集需要截断倒数的两个item (倒数第二个item作为valid，倒数第一个item作为test)
                item_seq = item_seq[:-1]
                for i in range(1, len(item_seq)):
                    history = item_seq[:i]
                    target = item_seq[i]
                    if len(history) > max_item_seq_len:
                        history = history[-max_item_seq_len:]
                    samples.append({
                        'user_id': user_id,
                        'history_items': history,
                        'target_item': target
                    })
            elif self.mode == 'valid':
                # 验证集：使用倒数第二个物品作为目标
                if len(item_seq) < 3:
                    continue
                history = item_seq[:-2]
                target = item_seq[-2]
                if len(history) > max_item_seq_len:
                    history = history[-max_item_seq_len:]
                samples.append({
                    'user_id': user_id,
                    'history_items': history,
                    'target_item': target
                })
            elif self.mode == 'test':
                # 测试集：使用最后一个物品作为目标
                if len(item_seq) < 2:
                    continue
                history = item_seq[:-1]
                target = item_seq[-1]
                if len(history) > max_item_seq_len:
                    history = history[-max_item_seq_len:]
                samples.append({
                    'user_id': user_id,
                    'history_items': history,
                    'target_item': target
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[int, List[int]]]:
        sample = self.samples[index]
        history_items = sample['history_items']
        target_item = sample['target_item']
        user_id     = sample['user_id']
        # 将历史物品转换为token序列（扁平化）
        source_tokens = []
        for item in history_items:
            if item in self.tokenizer.item2tokens:
                source_tokens.extend(self.tokenizer.item2tokens[item])
            else:
                # 如果物品不在tokenizer中，使用默认token（如0）
                source_tokens.extend([0] * self.tokens_per_item)
        
        # 将目标物品转换为token序列
        if target_item in self.tokenizer.item2tokens:
            target_tokens = self.tokenizer.item2tokens[target_item]
        else:
            target_tokens = [0] * self.tokens_per_item
        L_target = len(target_tokens)
        total_label_len = L_target + 1
        
        allowed_indices = []
        for i in range(total_label_len):
            #bos的next为digits1，digits1的next为digits2
            cycle_index = i % self.digits
            allowed_indices.append(self.cycle_masks[cycle_index])
        return {
            'user_token':    self.tokenizer.get_user_token(user_id),
            'source_tokens': source_tokens,
            'target_tokens': target_tokens,
            "target_id":     target_item,
            "allowed_indices": allowed_indices
        }