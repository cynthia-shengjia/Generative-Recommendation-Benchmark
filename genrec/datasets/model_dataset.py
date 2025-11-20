from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
from typing import Callable, Optional, Dict, List, Any, Tuple, Union
from genrec.tokenizers.GRTokenizer import AbstractTokenizer
import logging
import random
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
        
        # 新增：获取所有物品ID（用于负采样）
        if self.mode == 'offline-rl': 
            self.all_items = self._get_all_items()
            self.neg_num   = 4

        # 计算每个物品的token数量（假设所有物品相同）
        self.tokens_per_item = self._get_tokens_per_item()
        self.max_token_len = self.tokens_per_item * self.config['max_seq_len'] + 1
        # self._precompute_vocab_ranges_and_masks()

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
            
            # 2. [!change] 存储掩码的“构建块”
            
            # "第一个 token" (位置 0) 的掩码
            self.allowed_user_tokens = list(range(start_user, end_user))
            
            # "后续4个一循环" (位置 1, 2, 3, 4; 5, 6, 7, 8; ...) 的掩码
            self.cycle_masks = [
                list(range(start_digit_1, end_digit_1)),  # 循环 1
                list(range(start_digit_2, end_digit_2)),  # 循环 2
                list(range(start_digit_3, end_digit_3)),  # 循环 3
                list(range(start_dulicate, end_dulicate)), # 循环 4
            ]
            self.total_label_len = self.tokens_per_item + 1 

            logger.info("Dataset: Pre-converting cycle mask lists to tensors...")
            self.cycle_mask_tensors = []
            for ids_list in self.cycle_masks:
                if ids_list:
                    # 转换为长整型张量，用于索引
                    self.cycle_mask_tensors.append(torch.LongTensor(ids_list))
                else:
                    # 以 None 作为占位符
                    self.cycle_mask_tensors.append(None)
            
            # 3. 【核心】预计算静态的 loss_mask
            logger.info(
                f"Dataset: Pre-computing static loss mask of shape "
                f"[{self.total_label_len}, {self.vocab_size}]..."
            )
            
            # 创建一个全零的 mask
            mask = torch.zeros(
                self.total_label_len, 
                self.vocab_size, 
                dtype=torch.float
            )

            # 4. 填充这个 mask (这就是从 get_item 移过来的循环)
            #    (self.digits 是在 __init__ 中从 tokenizer 获取的)
            for i in range(self.total_label_len):
                # i 0 4
                # 0 1 2 3 0
                # 你的原始逻辑：i=0 -> cycle 0, i=1 -> cycle 1 ... i=4 -> cycle 0
                cycle_index = i % self.digits
                
                # 从预先转换的张量列表中获取
                allowed_tensor = self.cycle_mask_tensors[cycle_index] 
                
                if allowed_tensor is not None:
                    # 使用高效的张量索引一次性将所有位置设为 1.0
                    mask[i, allowed_tensor] = 1.0
            last_position_index = self.total_label_len - 1
            mask[last_position_index, :] = 0.0
            mask[last_position_index, self.tokenizer.eos_token] = 1.0
            # 5. 将最终的 mask 存储为类属性
            self.precomputed_loss_mask = mask
            logger.info("Dataset: Static loss mask pre-computation complete.")
    def _load_user_seqs(self) -> Dict[int, List[int]]:
        user_seqs = defaultdict(list)
        with open(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)
        for _, row in user_seqs_dataframe.iterrows():
            user_id = int(row['UserID'])
            item_seq = list(row["ItemID"])
            user_seqs[user_id] = item_seq
        return user_seqs
    

    def _get_all_items(self) -> List[int]:
        """获取所有物品ID"""
        all_items = set()
        for item_seq in self.user_seqs.values():
            all_items.update(item_seq)
        
        # 方法1: 使用实际出现的物品ID
        all_items_list = sorted(list(all_items))
        
        # 方法2: 如果你想使用 [0, max_item_id] 的连续范围
        # max_item_id = max(all_items) if all_items else 0
        # all_items_list = list(range(max_item_id + 1))
        
        return all_items_list


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
            elif self.mode == 'offline-rl':
                item_seq = item_seq[:-2]
                for i in range(1, len(item_seq)):
                    history = item_seq[:i]
                    target = item_seq[i]
                    
                    if len(history) > max_item_seq_len:
                        history = history[-max_item_seq_len:]
                    
                    user_interacted     = set(item_seq[:i+1])
                    candidate_negatives = [item for item in self.all_items if item not in user_interacted] 
                    negative_items = random.sample(
                        candidate_negatives, 
                        self.neg_num
                    )
                    
                    samples.append({
                        'user_id': user_id,
                        'history_items': history,
                        'target_item': target,
                        'negative_items': negative_items,  # 新增负样本
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

        result = {
            'user_token':    self.tokenizer.get_user_token(user_id),
            'source_tokens': source_tokens,
            'target_tokens': target_tokens,
            "target_id":     target_item,
            # "single_sample_loss_mask": self.precomputed_loss_mask,
        }

        if self.mode == "offline-rl":
            negative_items = sample['negative_items']
            
            rejected_tokens = {}
            

            for i, negative_item in enumerate(negative_items):
                if negative_item in self.tokenizer.item2tokens:
                    target_negative_tokens = self.tokenizer.item2tokens[negative_item]
                else:
                    target_negative_tokens = [0] * self.tokens_per_item
                rejected_tokens[i] = target_negative_tokens          
            
            result['rejected_tokens'] = rejected_tokens

        return result