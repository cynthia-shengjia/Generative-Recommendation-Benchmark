from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
from typing import Callable, Optional, Dict, List, Any, Tuple, Union

class HSTUDataset(Dataset):
    def __init__(
        self,
        data_interaction_files: str,
        config: dict,
        mode: str = 'train',  # 'train', 'valid', or 'test'
    ) -> None:
        self.config = config
        self.data_interaction_files = data_interaction_files
        self.mode = mode
        
        assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
        self.user_seqs = self._load_user_seqs()
        self.samples = self._create_samples()

    def _load_user_seqs(self) -> Dict[int, List[int]]:
        """加载用户交互序列"""
        user_seqs = defaultdict(lambda: ([], []))
        with open(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)
        for _, row in user_seqs_dataframe.iterrows():
            user_id = int(row['UserID'])
            item_seq = list(row["ItemID"])
            timestamp_seq = list(row["Timestamp"]) 
            user_seqs[user_id] = (item_seq, timestamp_seq)
        return user_seqs
    
    def _create_samples(self) -> List[Dict[str, List[int]]]:
        """
        为每个用户的序列创建多个训练样本。
        每个样本都是一个增长的序列前缀。
        例如：序列 [A, B, C, D] -> 生成样本 [A, B], [A, B, C], [A, B, C, D]
        """
        samples = []
        max_len = self.config['max_seq_len'] 

        for user_id, (item_seq, ts_seq) in self.user_seqs.items():
            if self.mode == 'train':
                train_item_seq = item_seq[:-2]
                train_ts_seq = ts_seq[:-2]
                if len(train_item_seq) < 2: continue
                # for i in range(2, len(train_item_seq)+1):
                #     # 目标是当前item
                #     input_seq = train_item_seq[:i]
                #     input_ts = train_ts_seq[:i]
                    
                #     # 截断输入序列，使其不超过 max_len
                #     input_seq = input_seq[-max_len:]
                #     input_ts = input_ts[-max_len:]
                    
                #     samples.append({
                #         'input_ids': input_seq,
                #         'timestamps': input_ts,
                #     })
                #HSTU不使用滑动窗口，先保留代码，试试效果
                seq = item_seq[:-2][-max_len:]
                ts = ts_seq[:-2][-max_len:]
                samples.append({
                    'input_ids': seq,
                    'timestamps': ts
                })
            elif self.mode == 'valid':
                # 验证集: 使用到倒数第二个 item 的序列
                if len(item_seq) < 3: continue
                seq = item_seq[:-1][-max_len:]
                ts = ts_seq[:-1][-max_len:]
                samples.append({
                    'input_ids': seq,
                    'timestamps': ts
                })

            elif self.mode == 'test':
                # 测试集: 使用完整的序列
                if len(item_seq) < 3: continue
                seq = item_seq[-max_len:]
                ts = ts_seq[-max_len:]
                samples.append({
                    'input_ids': seq,
                    'timestamps': ts
                })
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]
    
class SASRecDataset(Dataset):
    def __init__(
        self,
        data_interaction_files: str,
        config: dict,
        mode: str = 'train',  # 'train', 'valid', or 'test'
    ) -> None:
        self.config = config
        self.data_interaction_files = data_interaction_files
        self.mode = mode
        
        assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
        self.user_seqs = self._load_user_seqs()
        self.samples = self._create_samples()

    def _load_user_seqs(self) -> Dict[int, List[int]]:
        """加载用户交互序列"""
        user_seqs = defaultdict(lambda: ([], []))
        with open(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)
        for _, row in user_seqs_dataframe.iterrows():
            user_id = int(row['UserID'])
            item_seq = list(row["ItemID"])
            user_seqs[user_id] = item_seq
        return user_seqs
    
    def _create_samples(self) -> List[Dict[str, List[int]]]:
        """
        为每个用户的序列创建多个训练样本。
        每个样本都是一个增长的序列前缀。
        例如：序列 [A, B, C, D] -> 生成样本 [A, B], [A, B, C], [A, B, C, D]
        """
        samples = []
        max_len = self.config['max_seq_len'] 

        for user_id, item_seq in self.user_seqs.items():
            if self.mode == 'train':
                train_item_seq = item_seq[:-2]
                train_item_seq = train_item_seq[-max_len:]
                if len(train_item_seq) < 2: continue
                for i in range(2, len(train_item_seq)+1):
                    input_seq = train_item_seq[:i]
                    # input_seq = input_seq[-max_len:]
                    samples.append({
                        'input_ids': input_seq,
                    })
                # seq = train_item_seq[-max_len:]
                # samples.append({
                #     'input_ids': seq,
                # })
            elif self.mode == 'valid':
                # 验证集: 使用到倒数第二个 item 的序列
                if len(item_seq) < 3: continue
                seq = item_seq[:-1][-max_len:]
                samples.append({
                    'input_ids': seq,
                })

            elif self.mode == 'test':
                # 测试集: 使用完整的序列
                if len(item_seq) < 3: continue
                seq = item_seq[-max_len:]
                samples.append({
                    'input_ids': seq,
                })
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]
    
