from dataclasses import dataclass
from typing import Dict, List, Any
from genrec.tokenizers.TigerTokenizer import TigerTokenizer
import torch

@dataclass
class TrainSeqRecDataCollator:
    max_seq_len: int
    pad_token_id: int
    eos_token_id: int
    tokens_per_item: int = 4  # 假设每个物品有4个token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for feature in features:
            source_tokens = feature["source_tokens"]
            target_tokens = feature["target_tokens"]
            user_token       = feature['user_token']
            # 应用偏移到源序列（历史）
            transformed_source = [user_token] + source_tokens
            # 截断或填充源序列
            if len(transformed_source) > self.max_seq_len:
                transformed_source = transformed_source[-self.max_seq_len:]
            else:
                padding_length = self.max_seq_len - len(transformed_source)
                transformed_source = [self.pad_token_id] * padding_length + transformed_source
            input_ids_list.append(transformed_source)

            # 应用偏移到目标序列
            transformed_target = []
            for i, token in enumerate(target_tokens):
                transformed_target.append(token)
            transformed_target.append(self.eos_token_id)
            labels_list.append(transformed_target)

        # 填充标签序列，用-100忽略填充部分的损失
        max_label_len = max(len(lbl) for lbl in labels_list)
        for i in range(len(labels_list)):
            labels_list[i] = labels_list[i] + [-100] * (max_label_len - len(labels_list[i]))

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": (torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id).long(),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


@dataclass
class TestSeqRecDataCollator:
    max_seq_len: int
    pad_token_id: int
    eos_token_id: int
    tokens_per_item: int = 4  # 假设每个物品有4个token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for feature in features:
            source_tokens = feature["source_tokens"]
            target_tokens = feature["target_tokens"]
            user_token    = feature['user_token']
            label_id      = feature['target_id']
            
            # 应用偏移到源序列（历史）
            transformed_source = [user_token] + source_tokens
            # 截断或填充源序列
            if len(transformed_source) > self.max_seq_len:
                transformed_source = transformed_source[-self.max_seq_len:]
            else:
                padding_length = self.max_seq_len - len(transformed_source)
                transformed_source = [self.pad_token_id] * padding_length + transformed_source
            input_ids_list.append(transformed_source)
            labels_list.append(label_id)


        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": (torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id).long(),
            "label_id": labels_list
        }