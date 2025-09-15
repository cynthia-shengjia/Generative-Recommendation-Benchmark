import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import abc
import torch.nn.functional as F
import torch
import math
from typing import Dict, Tuple
# class HSTUSeqModelTrainingDataset(Dataset):
#     def __init__(
#         self,
#         data_interaction_files: str,
#         tokenizer: Any, # AbstractTokenizer,
#         config: dict,
#         mode: str = 'train',  # 'train', 'valid', or 'test'
#         device: Optional[str] = None
#     ) -> None:
#         self.config = config
#         self.data_interaction_files = data_interaction_files
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.device = device if device else torch.device('cpu')
        
#         assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
#         # CHANGED: _load_user_seqs 现在会同时加载物品和时间戳
#         self.user_seqs = self._load_user_seqs()
#         self.user_ids = list(self.user_seqs.keys())
        
#         self.tokens_per_item = self._get_tokens_per_item()
#         self.max_token_len = (self.tokens_per_item + 1) * self.config['max_seq_len'] + 1
        
#         # CHANGED: _create_samples 现在会同时处理物品和时间戳
#         self.samples = self._create_samples()


#     def _load_user_seqs(self) -> Dict[int, Dict[str, list]]:
#         """
#         加载用户序列数据，同时包含物品ID和时间戳。
        
#         **重要假设**: 输入的 pickle 文件是一个 DataFrame，
#         其中包含 'UserID', 'ItemID' (list), 和 'Timestamp' (list) 列。
#         """
#         user_seqs_data = defaultdict(dict)
#         with open(self.data_interaction_files, 'rb') as f:
#             user_seqs_dataframe = pickle.load(f)

#         # 确保时间戳列存在
#         if 'Timestamp' not in user_seqs_dataframe.columns:
#             raise ValueError("Interaction data file must contain a 'Timestamp' column.")

#         for _, row in user_seqs_dataframe.iterrows():
#             user_id = int(row['UserID'])
#             item_seq = list(row["ItemID"])
#             timestamp_seq = list(row["Timestamp"]) # NEW: 加载时间戳序列

#             if len(item_seq) != len(timestamp_seq):
#                 print(f"Warning: User {user_id} has mismatched item and timestamp sequence lengths. Skipping.")
#                 continue

#             user_seqs_data[user_id] = {
#                 'items': item_seq,
#                 'timestamps': timestamp_seq
#             }
#         return user_seqs_data
    
#     def _get_tokens_per_item(self) -> int:
#         if not hasattr(self.tokenizer, 'item2tokens') or not self.tokenizer.item2tokens:
#             return 1
#         first_item = next(iter(self.tokenizer.item2tokens.keys()))
#         return len(self.tokenizer.item2tokens[first_item])

#     # CHANGED: 此函数现在同步处理物品和时间戳序列
#     def _create_samples(self) -> List[Dict[str, Any]]:
#         """创建样本，返回历史物品/时间戳序列和目标物品"""
#         samples = []
#         max_item_seq_len = self.config['max_seq_len']
        
#         for user_id, seq_data in self.user_seqs.items():
#             item_seq = seq_data['items']
#             timestamp_seq = seq_data['timestamps'] # NEW: 获取时间戳序列

#             if self.mode == 'train':
#                 # 训练集: item_seq[:-2], timestamp_seq[:-2]
#                 train_item_seq = item_seq[:-2]
#                 train_timestamp_seq = timestamp_seq[:-2]
#                 for i in range(1, len(train_item_seq)):
#                     history = train_item_seq[:i]
#                     history_ts = train_timestamp_seq[:i] # NEW: 同步切分时间戳
#                     target = train_item_seq[i]
                    
#                     if len(history) > max_item_seq_len:
#                         history = history[-max_item_seq_len:]
#                         history_ts = history_ts[-max_item_seq_len:] # NEW: 同步截断时间戳

#                     samples.append({
#                         'user_id': user_id,
#                         'history_items': history,
#                         'history_timestamps': history_ts, # NEW: 保存历史时间戳
#                         'target_item': target
#                     })

#             elif self.mode == 'valid':
#                 if len(item_seq) < 3: continue
#                 history = item_seq[:-2]
#                 history_ts = timestamp_seq[:-2] # NEW
#                 target = item_seq[-2]
#                 if len(history) > max_item_seq_len:
#                     history = history[-max_item_seq_len:]
#                     history_ts = history_ts[-max_item_seq_len:] # NEW
#                 samples.append({
#                     'user_id': user_id,
#                     'history_items': history,
#                     'history_timestamps': history_ts, # NEW
#                     'target_item': target
#                 })

#             elif self.mode == 'test':
#                 if len(item_seq) < 2: continue
#                 history = item_seq[:-1]
#                 history_ts = timestamp_seq[:-1] # NEW
#                 target = item_seq[-1]
#                 if len(history) > max_item_seq_len:
#                     history = history[-max_item_seq_len:]
#                     history_ts = history_ts[-max_item_seq_len:] # NEW
#                 samples.append({
#                     'user_id': user_id,
#                     'history_items': history,
#                     'history_timestamps': history_ts, # NEW
#                     'target_item': target
#                 })
#         return samples

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> Dict[str, Union[int, List[int]]]:
#         """
#         返回一个独立的、未经处理的样本。
#         将所有 tokenization, padding, 和 batching 逻辑留给 Data Collator。
#         """
#         sample = self.samples[index]
        
#         # 直接返回原始数据，DataCollator 会处理剩下的事情
#         return {
#             'user_id': sample['user_id'],
#             'history_items': sample['history_items'],
#             'history_timestamps': sample['history_timestamps'],
#             'target_item': sample['target_item']
#         }


class HSTUSeqModelTrainingDataset:
    """Decoder-Only Dataset"""
    def __init__(
        self,
        data_interaction_files: str,
        tokenizer: Any,
        config: dict,
        mode: str = 'train',
        device: Optional[str] = None
    ) -> None:
        self.config = config
        self.data_interaction_files = data_interaction_files
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device if device else torch.device('cpu')
        
        assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
        self.user_seqs = self._load_user_seqs()
        self.user_ids = list(self.user_seqs.keys())
        
        self.tokens_per_item = self._get_tokens_per_item()
        self.max_token_len = self.tokens_per_item * self.config['max_seq_len'] + 2
        
        self.samples = self._create_samples()

    def _load_user_seqs(self) -> Dict[int, Dict[str, list]]:
        """加载用户序列数据，包含物品ID和时间戳"""
        user_seqs_data = defaultdict(dict)
        with open(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)

        if 'Timestamp' not in user_seqs_dataframe.columns:
            raise ValueError("Interaction data file must contain a 'Timestamp' column.")

        for _, row in user_seqs_dataframe.iterrows():
            user_id = int(row['UserID'])
            item_seq = list(row["ItemID"])
            timestamp_seq = list(row["Timestamp"])

            if len(item_seq) != len(timestamp_seq):
                print(f"Warning: User {user_id} has mismatched item and timestamp sequence lengths. Skipping.")
                continue

            user_seqs_data[user_id] = {
                'items': item_seq,
                'timestamps': timestamp_seq
            }
        return user_seqs_data
    
    def _get_tokens_per_item(self) -> int:
        if not hasattr(self.tokenizer, 'item2tokens') or not self.tokenizer.item2tokens:
            return 1
        first_item = next(iter(self.tokenizer.item2tokens.keys()))
        return len(self.tokenizer.item2tokens[first_item])

    def _create_samples(self) -> List[Dict[str, Any]]:
        """创建decoder-only样本"""
        samples = []
        max_item_seq_len = self.config['max_seq_len']
        
        for user_id, seq_data in self.user_seqs.items():
            item_seq = seq_data['items']
            timestamp_seq = seq_data['timestamps']

            if self.mode == 'train':
                # 训练：使用滑动窗口创建多个样本
                train_item_seq = item_seq[:-2]
                train_timestamp_seq = timestamp_seq[:-2]
                
                for i in range(1, len(train_item_seq)):
                    full_sequence = train_item_seq[:i+1]  # 包括target
                    full_timestamps = train_timestamp_seq[:i+1]
                    
                    if len(full_sequence) > max_item_seq_len:
                        full_sequence = full_sequence[-max_item_seq_len:]
                        full_timestamps = full_timestamps[-max_item_seq_len:]

                    samples.append({
                        'user_id': user_id,
                        'sequence_items': full_sequence,
                        'sequence_timestamps': full_timestamps,
                    })

            elif self.mode == 'valid':
                if len(item_seq) < 3: continue
                full_sequence = item_seq[:-1]  # 不包括最后一个test item
                full_timestamps = timestamp_seq[:-1]
                
                if len(full_sequence) > max_item_seq_len:
                    full_sequence = full_sequence[-max_item_seq_len:]
                    full_timestamps = full_timestamps[-max_item_seq_len:]
                    
                samples.append({
                    'user_id': user_id,
                    'sequence_items': full_sequence,
                    'sequence_timestamps': full_timestamps,
                    'target_item': item_seq[-2],  # 验证目标
                })

            elif self.mode == 'test':
                if len(item_seq) < 2: continue
                full_sequence = item_seq[:-1]  # 历史序列
                full_timestamps = timestamp_seq[:-1]
                
                if len(full_sequence) > max_item_seq_len:
                    full_sequence = full_sequence[-max_item_seq_len:]
                    full_timestamps = full_timestamps[-max_item_seq_len:]
                    
                samples.append({
                    'user_id': user_id,
                    'sequence_items': full_sequence,
                    'sequence_timestamps': full_timestamps,
                    'target_item': item_seq[-1],  # 测试目标
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """返回一个样本"""
        sample = self.samples[index]
        return sample


@dataclass
class HSTUDecoderOnlyDataCollator:
    """Decoder-Only Data Collator for training with right padding"""
    tokenizer: Any
    max_token_len: int
    ignore_index: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        timestamps_list = []
        labels_list = []
        
        pad_token_id = self.tokenizer.pad_token
        tokens_per_item = self._get_tokens_per_item()

        for feature in features:
            sequence_items = feature["sequence_items"]
            sequence_timestamps = feature["sequence_timestamps"]

            # 转换物品序列为token序列
            sequence_tokens = []
            sequence_aligned_timestamps = []
            
            for item, timestamp in zip(sequence_items, sequence_timestamps):
                if item in self.tokenizer.item2tokens:
                    item_tokens = self.tokenizer.item2tokens[item]
                    sequence_tokens.extend(item_tokens)
                    sequence_aligned_timestamps.extend([timestamp] * len(item_tokens))
                else:
                    sequence_tokens.extend([pad_token_id] * tokens_per_item)
                    sequence_aligned_timestamps.extend([0] * tokens_per_item)
            
            bos_timestamp = sequence_aligned_timestamps[0] if sequence_aligned_timestamps else 0
            eos_timestamp = sequence_aligned_timestamps[-1] if sequence_aligned_timestamps else 0
            full_input_tokens = [self.tokenizer.bos_token] + sequence_tokens + [self.tokenizer.eos_token]
            full_timestamps = [bos_timestamp] + sequence_aligned_timestamps + [eos_timestamp]
            
            
            if len(full_input_tokens) > self.max_token_len:
                available_len = self.max_token_len - 2  # 减去BOS和EOS的位置
                truncated_sequence = sequence_tokens[-available_len:]
                truncated_timestamps = sequence_aligned_timestamps[-available_len:]
                final_input_tokens = [self.tokenizer.bos_token] + truncated_sequence + [self.tokenizer.eos_token]
                final_timestamps = [bos_timestamp] + truncated_timestamps + [eos_timestamp]
            else:
                # 需要填充 - 改为右填充 (Right Padding)
                padding_len = self.max_token_len - len(full_input_tokens)
                final_input_tokens = full_input_tokens + [pad_token_id] * padding_len
                final_timestamps = full_timestamps + [0] * padding_len

            input_ids_list.append(final_input_tokens)
            timestamps_list.append(final_timestamps)

            # 生成labels：输入序列向右偏移一位作为目标
            labels = final_input_tokens[1:] + [self.ignore_index]
            
            # 对于padding位置和BOS位置，labels设为ignore_index
            labels = [
                self.ignore_index if (input_token == pad_token_id or i == 0) else label
                for i, (input_token, label) in enumerate(zip(final_input_tokens, labels))
            ]
            labels_list.append(labels)

        # 转换为tensor
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = (input_ids != pad_token_id).long()
        labels = torch.tensor(labels_list, dtype=torch.long)
        timestamps = torch.tensor(timestamps_list, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "past_payloads": {
                "timestamps": timestamps
            }
        }

    def _get_tokens_per_item(self) -> int:
        if not hasattr(self.tokenizer, 'item2tokens') or not self.tokenizer.item2tokens:
            return 1
        first_item = next(iter(self.tokenizer.item2tokens.keys()))
        return len(self.tokenizer.item2tokens[first_item])

    
# class HSTUSeqModelTrainingDataset:
#     """Decoder-Only Dataset"""
#     def __init__(
#         self,
#         data_interaction_files: str,
#         tokenizer: Any,
#         config: dict,
#         mode: str = 'train',
#         device: Optional[str] = None
#     ) -> None:
#         self.config = config
#         self.data_interaction_files = data_interaction_files
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.device = device if device else torch.device('cpu')
        
#         assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
#         self.user_seqs = self._load_user_seqs()
#         self.user_ids = list(self.user_seqs.keys())
        
#         self.tokens_per_item = self._get_tokens_per_item()
#         self.max_token_len = self.tokens_per_item * self.config['max_seq_len'] + 2
        
#         self.samples = self._create_samples()

#     def _load_user_seqs(self) -> Dict[int, Dict[str, list]]:
#         """加载用户序列数据，包含物品ID和时间戳"""
#         user_seqs_data = defaultdict(dict)
#         with open(self.data_interaction_files, 'rb') as f:
#             user_seqs_dataframe = pickle.load(f)

#         if 'Timestamp' not in user_seqs_dataframe.columns:
#             raise ValueError("Interaction data file must contain a 'Timestamp' column.")

#         for _, row in user_seqs_dataframe.iterrows():
#             user_id = int(row['UserID'])
#             item_seq = list(row["ItemID"])
#             timestamp_seq = list(row["Timestamp"])

#             if len(item_seq) != len(timestamp_seq):
#                 print(f"Warning: User {user_id} has mismatched item and timestamp sequence lengths. Skipping.")
#                 continue

#             user_seqs_data[user_id] = {
#                 'items': item_seq,
#                 'timestamps': timestamp_seq
#             }
#         return user_seqs_data
    
#     def _get_tokens_per_item(self) -> int:
#         if not hasattr(self.tokenizer, 'item2tokens') or not self.tokenizer.item2tokens:
#             return 1
#         first_item = next(iter(self.tokenizer.item2tokens.keys()))
#         return len(self.tokenizer.item2tokens[first_item])

#     def _create_samples(self) -> List[Dict[str, Any]]:
#         """创建decoder-only样本"""
#         samples = []
#         max_item_seq_len = self.config['max_seq_len']
        
#         for user_id, seq_data in self.user_seqs.items():
#             item_seq = seq_data['items']
#             timestamp_seq = seq_data['timestamps']

#             if self.mode == 'train':
#                 # 训练：使用滑动窗口创建多个样本
#                 train_item_seq = item_seq[:-2]
#                 train_timestamp_seq = timestamp_seq[:-2]
                
#                 for i in range(1, len(train_item_seq)):
#                     full_sequence = train_item_seq[:i+1]  # 包括target
#                     full_timestamps = train_timestamp_seq[:i+1]
                    
#                     if len(full_sequence) > max_item_seq_len:
#                         full_sequence = full_sequence[-max_item_seq_len:]
#                         full_timestamps = full_timestamps[-max_item_seq_len:]

#                     samples.append({
#                         'user_id': user_id,
#                         'sequence_items': full_sequence,
#                         'sequence_timestamps': full_timestamps,
#                     })

#             elif self.mode == 'valid':
#                 if len(item_seq) < 3: continue
#                 full_sequence = item_seq[:-1]  # 不包括最后一个test item
#                 full_timestamps = timestamp_seq[:-1]
                
#                 if len(full_sequence) > max_item_seq_len:
#                     full_sequence = full_sequence[-max_item_seq_len:]
#                     full_timestamps = full_timestamps[-max_item_seq_len:]
                    
#                 samples.append({
#                     'user_id': user_id,
#                     'sequence_items': full_sequence,
#                     'sequence_timestamps': full_timestamps,
#                     'target_item': item_seq[-2],  # 验证目标
#                 })

#             elif self.mode == 'test':
#                 if len(item_seq) < 2: continue
#                 full_sequence = item_seq[:-1]  # 历史序列
#                 full_timestamps = timestamp_seq[:-1]
                
#                 if len(full_sequence) > max_item_seq_len:
#                     full_sequence = full_sequence[-max_item_seq_len:]
#                     full_timestamps = full_timestamps[-max_item_seq_len:]
                    
#                 samples.append({
#                     'user_id': user_id,
#                     'sequence_items': full_sequence,
#                     'sequence_timestamps': full_timestamps,
#                     'target_item': item_seq[-1],  # 测试目标
#                 })
#         return samples

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> Dict[str, Any]:
#         """返回一个样本"""
#         sample = self.samples[index]
#         return sample


# @dataclass
# class HSTUDecoderOnlyDataCollator:
#     """Decoder-Only Data Collator for training"""
#     tokenizer: Any
#     max_token_len: int
#     ignore_index: int = -100

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         input_ids_list = []
#         timestamps_list = []
#         labels_list = []
        
#         pad_token_id = self.tokenizer.pad_token
#         tokens_per_item = self._get_tokens_per_item()

#         for feature in features:
#             sequence_items = feature["sequence_items"]
#             sequence_timestamps = feature["sequence_timestamps"]

#             # 转换物品序列为token序列
#             sequence_tokens = []
#             sequence_aligned_timestamps = []
            
#             for item, timestamp in zip(sequence_items, sequence_timestamps):
#                 if item in self.tokenizer.item2tokens:
#                     item_tokens = self.tokenizer.item2tokens[item]
#                     sequence_tokens.extend(item_tokens)
#                     sequence_aligned_timestamps.extend([timestamp] * len(item_tokens))
#                 else:
#                     sequence_tokens.extend([pad_token_id] * tokens_per_item)
#                     sequence_aligned_timestamps.extend([0] * tokens_per_item)
            
#             bos_timestamp = sequence_aligned_timestamps[0] if sequence_aligned_timestamps else 0
#             eos_timestamp = sequence_aligned_timestamps[-1] if sequence_aligned_timestamps else 0
#             full_input_tokens = [self.tokenizer.bos_token] + sequence_tokens + [self.tokenizer.eos_token]
#             full_timestamps = [bos_timestamp] + sequence_aligned_timestamps + [eos_timestamp]
            
            
#             if len(full_input_tokens) > self.max_token_len:
#                 # 保留BOS，从序列中间截断，保留EOS
#                 available_len = self.max_token_len - 2  # 减去BOS和EOS的位置
                
#                 truncated_sequence = sequence_tokens[-available_len:]
#                 truncated_timestamps = sequence_aligned_timestamps[-available_len:]
#                 final_input_tokens = [self.tokenizer.bos_token] + truncated_sequence + [self.tokenizer.eos_token]
#                 final_timestamps = [bos_timestamp] + truncated_timestamps + [eos_timestamp]
#             else:
#                 # 需要填充
#                 padding_len = self.max_token_len - len(full_input_tokens)
#                 final_input_tokens = [pad_token_id] * padding_len + full_input_tokens
#                 final_timestamps = [0] * padding_len + full_timestamps

#             input_ids_list.append(final_input_tokens)
#             timestamps_list.append(final_timestamps)

#             # 修复：最后一个位置应该用ignore_index而不是pad_token_id
#             labels = final_input_tokens[1:] + [self.ignore_index]
            
#             # 对于padding位置和BOS位置，labels设为ignore_index
#             labels = [
#                 self.ignore_index if (input_token == pad_token_id or i == 0) else label
#                 for i, (input_token, label) in enumerate(zip(final_input_tokens, labels))
#             ]
#             labels_list.append(labels)

#         # 转换为tensor
#         input_ids = torch.tensor(input_ids_list, dtype=torch.long)
#         attention_mask = (input_ids != pad_token_id).long()
#         labels = torch.tensor(labels_list, dtype=torch.long)
#         timestamps = torch.tensor(timestamps_list, dtype=torch.long)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels,
#             "past_payloads": {
#                 "timestamps": timestamps
#             }
#         }

#     def _get_tokens_per_item(self) -> int:
#         if not hasattr(self.tokenizer, 'item2tokens') or not self.tokenizer.item2tokens:
#             return 1
#         first_item = next(iter(self.tokenizer.item2tokens.keys()))
#         return len(self.tokenizer.item2tokens[first_item])


@dataclass
class HSTUDecoderOnlyInferenceCollator:
    """推理阶段的Data Collator"""
    tokenizer: Any
    max_token_len: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        timestamps_list = []
        target_items_list = []
        
        pad_token_id = self.tokenizer.pad_token
        tokens_per_item = self._get_tokens_per_item()
        vocab_size = self.tokenizer.vocab_size

        for feature in features:
            sequence_items = feature["sequence_items"]
            sequence_timestamps = feature["sequence_timestamps"]

            # 转换物品序列为token序列
            sequence_tokens = []
            sequence_aligned_timestamps = []
            
            for item, timestamp in zip(sequence_items, sequence_timestamps):
                if item in self.tokenizer.item2tokens:
                    item_tokens = self.tokenizer.item2tokens[item]
                    # 验证每个item token是否在合理范围内
                    valid_item_tokens = []
                    for token in item_tokens:
                        if 0 <= token < vocab_size:
                            valid_item_tokens.append(token)
                        else:
                            print(f"Warning: item token {token} out of range, using pad_token_id")
                            valid_item_tokens.append(pad_token_id)
                    
                    sequence_tokens.extend(valid_item_tokens)
                    sequence_aligned_timestamps.extend([timestamp] * len(valid_item_tokens))
                else:
                    sequence_tokens.extend([pad_token_id] * tokens_per_item)
                    sequence_aligned_timestamps.extend([0] * tokens_per_item)
            
            # 构建输入：BOS + sequence（推理时不加EOS，让模型生成）
            bos_timestamp = sequence_aligned_timestamps[0] if sequence_aligned_timestamps else 0
            full_input_tokens = [self.tokenizer.bos_token] + sequence_tokens
            full_timestamps = [bos_timestamp] + sequence_aligned_timestamps

            # 截断或填充
            if len(full_input_tokens) > self.max_token_len:
                # 保留BOS，从序列末尾截取
                available_len = self.max_token_len - 1  # 减去BOS的位置
                truncated_sequence = sequence_tokens[-available_len:]
                truncated_timestamps = sequence_aligned_timestamps[-available_len:]
                final_input_tokens = [self.tokenizer.bos_token] + truncated_sequence
                final_timestamps = [bos_timestamp] + truncated_timestamps
            else:
                # 需要填充
                padding_len = self.max_token_len - len(full_input_tokens)
                final_input_tokens = full_input_tokens + [pad_token_id] * padding_len
                final_timestamps = full_timestamps + [0] * padding_len
                # padding_len = self.max_token_len - len(full_input_tokens)
                # final_input_tokens = [pad_token_id] * padding_len + full_input_tokens
                # final_timestamps = [0] * padding_len + full_timestamps

            # 最终验证：确保所有token都在合理范围内
            final_input_tokens = [max(0, min(token, vocab_size - 1)) for token in final_input_tokens]
            
            input_ids_list.append(final_input_tokens)
            timestamps_list.append(final_timestamps)
            
            # 保存目标物品用于评估
            if 'target_item' in feature:
                target_items_list.append(feature['target_item'])

        # 转换为tensor
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = (input_ids != pad_token_id).long()
        timestamps = torch.tensor(timestamps_list, dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_payloads": {
                "timestamps": timestamps
            }
        }
        
        if target_items_list:
            result["target_ids"] = torch.tensor(target_items_list, dtype=torch.long)
            
        return result

    def _get_tokens_per_item(self) -> int:
        if not hasattr(self.tokenizer, 'item2tokens') or not self.tokenizer.item2tokens:
            return 1
        first_item = next(iter(self.tokenizer.item2tokens.keys()))
        return len(self.tokenizer.item2tokens[first_item])
def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x

class EmbeddingModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
class OutputPostprocessorModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pass


class L2NormEmbeddingPostprocessor(OutputPostprocessorModule):

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._eps: float = eps

    def debug_str(self) -> str:
        return "l2"

    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        return output_embeddings / torch.clamp(
            torch.linalg.norm(output_embeddings, ord=None, dim=-1, keepdim=True),
            min=self._eps,
        )


class LayerNormEmbeddingPostprocessor(OutputPostprocessorModule):

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._eps: float = eps

    def debug_str(self) -> str:
        return "ln"

    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        return F.layer_norm(
            output_embeddings,
            normalized_shape=(self._embedding_dim,),
            eps=self._eps,
        )
class InputFeaturesPreprocessorModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):

    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):

    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim + rating_embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"posir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()

        user_embeddings = torch.cat(
            [past_embeddings, self._rating_emb(past_payloads["ratings"].int())],
            dim=-1,
        ) * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):

    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        self._rating_embedding_dim: int = rating_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_payloads["ratings"].int()),
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (
            self.get_preprocessed_masks(
                past_lengths,
                past_ids,
                past_embeddings,
                past_payloads,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask
class InteractionModule(torch.nn.Module):

    @abc.abstractmethod
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_item_sideinfo(
        self,
        item_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        pass

    @abc.abstractmethod
    def interaction(
        self,
        input_embeddings: torch.Tensor,  # [B, D]
        target_ids: torch.Tensor,  # [1, X] or [B, X]
        target_embeddings: Optional[torch.Tensor] = None,  # [1, X, D'] or [B, X, D']
    ) -> torch.Tensor:
        pass
class NDPModule(torch.nn.Module):

    def forward(  # pyre-ignore[3]
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_sideinfo: Optional[torch.Tensor],
        item_ids: torch.Tensor,
        precomputed_logits: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_embeddings: (B, input_embedding_dim) x float
            item_embeddings: (1/B, X, item_embedding_dim) x float
            item_sideinfo: (1/B, X, item_sideinfo_dim) x float

        Returns:
            Tuple of (B, X,) similarity values, keyed outputs
        """
        pass

class GeneralizedInteractionModule(InteractionModule):
    def __init__(
        self,
        ndp_module: NDPModule,
    ) -> None:
        super().__init__()

        self._ndp_module: NDPModule = ndp_module

    @abc.abstractmethod
    def debug_str(
        self,
    ) -> str:
        pass

    def interaction(
        self,
        input_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch._assert(
            len(input_embeddings.size()) == 2, "len(input_embeddings.size()) must be 2"
        )
        torch._assert(len(target_ids.size()) == 2, "len(target_ids.size()) must be 2")
        if target_embeddings is None:
            target_embeddings = self.get_item_embeddings(target_ids)
        torch._assert(
            len(target_embeddings.size()) == 3,
            "len(target_embeddings.size()) must be 3",
        )

        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            return self._ndp_module(
                input_embeddings=input_embeddings,  # [B, self._input_embedding_dim]
                item_embeddings=target_embeddings,  # [1/B, X, self._item_embedding_dim]
                item_sideinfo=self.get_item_sideinfo(
                    item_ids=target_ids
                ),  # [1/B, X, self._item_sideinfo_dim]
                item_ids=target_ids,
                precomputed_logits=None,
            )
def get_current_embeddings(
    lengths: torch.Tensor,
    encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int
        seq_embeddings: (B, N, D,) x float

    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.size()
    flattened_offsets = (lengths - 1) + torch.arange(
        start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
    ) * N
    return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)

class InteractionModule(torch.nn.Module):

    @abc.abstractmethod
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_item_sideinfo(
        self,
        item_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        pass

    @abc.abstractmethod
    def interaction(
        self,
        input_embeddings: torch.Tensor,  # [B, D]
        target_ids: torch.Tensor,  # [1, X] or [B, X]
        target_embeddings: Optional[torch.Tensor] = None,  # [1, X, D'] or [B, X, D']
    ) -> torch.Tensor:
        pass


class GeneralizedInteractionModule(InteractionModule):
    def __init__(
        self,
        ndp_module: NDPModule,
    ) -> None:
        super().__init__()

        self._ndp_module: NDPModule = ndp_module

    @abc.abstractmethod
    def debug_str(
        self,
    ) -> str:
        pass

    def interaction(
        self,
        input_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch._assert(
            len(input_embeddings.size()) == 2, "len(input_embeddings.size()) must be 2"
        )
        torch._assert(len(target_ids.size()) == 2, "len(target_ids.size()) must be 2")
        if target_embeddings is None:
            target_embeddings = self.get_item_embeddings(target_ids)
        torch._assert(
            len(target_embeddings.size()) == 3,
            "len(target_embeddings.size()) must be 3",
        )

        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            return self._ndp_module(
                input_embeddings=input_embeddings,  # [B, self._input_embedding_dim]
                item_embeddings=target_embeddings,  # [1/B, X, self._item_embedding_dim]
                item_sideinfo=self.get_item_sideinfo(
                    item_ids=target_ids
                ),  # [1/B, X, self._item_sideinfo_dim]
                item_ids=target_ids,
                precomputed_logits=None,
            )
        
class DotProductSimilarity(NDPModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def debug_str(self) -> str:
        return "dp"

    def forward(  # pyre-ignore [3]
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_sideinfo: Optional[torch.Tensor],
        item_ids: torch.Tensor,
        precomputed_logits: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_embeddings: (B, D,) or (B * r, D) x float.
            item_embeddings: (1, X, D) or (B, X, D) x float.

        Returns:
            (B, X) x float (or (B * r, X) x float).
        """
        del item_ids

        if item_embeddings.size(0) == 1:
            # [B, D] x ([1, X, D] -> [D, X]) => [B, X]
            return (
                torch.mm(input_embeddings, item_embeddings.squeeze(0).t()),
                {},
            )  # [B, X]
        elif input_embeddings.size(0) != item_embeddings.size(0):
            # (B * r, D) x (B, X, D).
            B, X, D = item_embeddings.size()
            return torch.bmm(
                input_embeddings.view(B, -1, D), item_embeddings.permute(0, 2, 1)
            ).view(-1, X)
        else:
            # assert input_embeddings.size(0) == item_embeddings.size(0)
            # [B, X, D] x ([B, D] -> [B, D, 1]) => [B, X, 1] -> [B, X]
            return torch.bmm(item_embeddings, input_embeddings.unsqueeze(2)).squeeze(2)