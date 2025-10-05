from dataclasses import dataclass
from typing import Dict, List, Any
from genrec.tokenizers.TigerTokenizer import TigerTokenizer
import torch

@dataclass
class HSTUDataCollator:

    def __init__(self, pad_token_id: int=0, pad_timestamp_value: int=0, max_seq_len: int=20):
        self.pad_token_id = pad_token_id
        self.pad_timestamp_value = pad_timestamp_value
        self.max_seq_len = max_seq_len
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        input_sequences = []
        label_sequences = []
        ts_sequences_input = []
        for f in features:
            input_sequences.append(f["input_ids"][:-1])
            label_sequences.append(f["input_ids"][1:])
            ts_sequences_input.append(f["timestamps"][:-1])
        max_length = self.max_seq_len - 1
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        padded_timestamps = []
        for i in range(len(input_sequences)):
            ids = input_sequences[i]
            labels = label_sequences[i]
            ts = ts_sequences_input[i]
            
            # 断言确保输入和标签长度一致
            assert len(ids) == len(labels) == len(ts)

            padding_length = max_length - len(ids)
            
            # Padding `input_ids` 和 `timestamps` (右padding)
            padded_ids = ids + [self.pad_token_id] * padding_length
            padded_ts = ts + [self.pad_timestamp_value] * padding_length
            
            # 创建 attention_mask
            mask = [1] * len(ids) + [0] * padding_length
            
            # Padding `labels`
            padded_label = labels + [-100] * padding_length
            
            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_label)
            attention_masks.append(mask)
            padded_timestamps.append(padded_ts)

        # 3. 转换为 PyTorch Tensors
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "timestamps": torch.tensor(padded_timestamps, dtype=torch.long),
        }
        
        return batch
    
@dataclass
class SASRecDataCollator:

    def __init__(self, pad_token_id: int=0, pad_timestamp_value: int=0, max_seq_len: int=20):
        self.pad_token_id = pad_token_id
        self.pad_timestamp_value = pad_timestamp_value
        self.max_seq_len = max_seq_len
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        input_sequences = []
        label_sequences = []
        for f in features:
            input_sequences.append(f["input_ids"][:-1])
            label_sequences.append(f["input_ids"][1:])
        max_length = self.max_seq_len - 1
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        for i in range(len(input_sequences)):
            ids = input_sequences[i]
            labels = label_sequences[i]
            

            padding_length = max_length - len(ids)
            #TODO：SASRec是左padding
            # Padding `input_ids` 和 `timestamps` (右padding)
            padded_ids = [self.pad_token_id] * padding_length + ids
            
            mask =  [0] * padding_length + [1] * len(ids)
            
            # Padding `labels`
            padded_label =  [-100] * padding_length + labels
            
            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_label)
            attention_masks.append(mask)

        # 3. 转换为 PyTorch Tensors
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        
        return batch