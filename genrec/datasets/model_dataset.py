from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
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
        
        # 加载数据
        self.item_reviews = self._load_item_reviews()
        self.user_seqs = self._load_user_seqs()
        self.user_ids = list(self.user_seqs.keys())
        
        # 计算每个物品的token数量（假设所有物品相同）
        self.tokens_per_item = self._get_tokens_per_item()
        self.max_token_len = (self.tokens_per_item + 1) * self.config['max_seq_len'] + 1
        
        # 直接创建样本，而不是先预处理整个序列
        self.samples = self._create_samples()

    def _load_item_reviews(self) -> Dict[int, str]:
        # 保持不变
        item_reviews = defaultdict(str)
        with open(self.data_text_files, 'rb') as f:
            item_titles_dataframe = pickle.load(f)
        for _, row in item_titles_dataframe.iterrows():
            item_id = int(row['ItemID'])
            item_context_info = row['Title']
            item_reviews[item_id] = item_context_info
        return item_reviews

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
        
        return {
            'user_token':    self.tokenizer.get_user_token(user_id),
            'source_tokens': source_tokens,
            'target_tokens': target_tokens,
            "target_id":     target_item
        }