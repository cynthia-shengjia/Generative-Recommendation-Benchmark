from dataclasses import dataclass
from typing import Dict, List, Any
from genrec.tokenizers.TigerTokenizer import TigerTokenizer
import torch

@dataclass
class TrainSeqRecDataCollator:
    max_seq_len: int
    pad_token_id: int
    eos_token_id: int
    vocab_size: int
    tokens_per_item: int = 4  # 假设每个物品有4个token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        target_allowed_indices_list = []
        has_vocab_mask_info = "target_allowed_indices" in features[0]
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

            if has_vocab_mask_info:
                allowed_indices = feature.get("target_allowed_indices")
                
                if allowed_indices is None or len(allowed_indices) != len(transformed_target):
                    raise ValueError(
                        f"Dataset 错误：'target_allowed_indices' 的长度 "
                        f"({len(allowed_indices) if allowed_indices else 'None'}) "
                        f"与 'target_tokens' + EOS 的长度 ({len(transformed_target)}) 不匹配。"
                    )
                target_allowed_indices_list.append(allowed_indices)
        # 填充标签序列，用-100忽略填充部分的损失
        max_label_len = max(len(lbl) for lbl in labels_list)
        for i in range(len(labels_list)):
            labels_list[i] = labels_list[i] + [-100] * (max_label_len - len(labels_list[i]))
        # 添加 loss_mask，1代表可计算loss，0代表mask
        batch_size = len(features)
        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": (torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id).long(),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
        if has_vocab_mask_info:
            # 1. 初始化一个全 0 的 mask
            #    形状: [batch_size, max_label_len, vocab_size]
            loss_mask = torch.zeros(
                batch_size, 
                max_label_len, 
                self.vocab_size,
                dtype=torch.float
            )
            
            # 2. 遍历批次中的每个序列
            for i in range(batch_size):
                # 3. 遍历序列中的每个 *真实* (非填充) token
                unpadded_len = unpadded_label_lengths[i]
                for j in range(unpadded_len):
                    
                    # 4. 获取为这个 [i, j] token 允许的 vocab ID 列表
                    allowed_vocab_ids = target_allowed_indices_list[i][j]
                    
                    # 5. 如果列表不为空，则将这些位置的 mask 设置为 1.0
                    if allowed_vocab_ids:
                        # 我们使用 torch.LongTensor 进行高级索引
                        loss_mask[i, j, torch.LongTensor(allowed_vocab_ids)] = 1.0
            
            # 6. 将最终的 mask 添加到批次中
            batch["loss_mask"] = loss_mask
        return batch
        # return {
        #     "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        #     "attention_mask": (torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id).long(),
        #     "labels": torch.tensor(labels_list, dtype=torch.long),
        # }


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