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
        loss_mask_list = []
        unpadded_label_lengths = []
        target_allowed_indices_list = []
        has_vocab_mask_info = "single_sample_loss_mask" in features[0]
        for feature in features:
            source_tokens = feature["source_tokens"]
            target_tokens = feature["target_tokens"]
            user_token       = feature['user_token']
            # 应用偏移到源序列（历史）
            transformed_source = source_tokens
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
            unpadded_label_lengths.append(len(transformed_target))
            if has_vocab_mask_info:
                # 直接获取预先计算好的 mask
                loss_mask_list.append(feature["single_sample_loss_mask"])
            # if has_vocab_mask_info:
            #     allowed_indices = feature.get("allowed_indices")
                
            #     if allowed_indices is None or len(allowed_indices) != len(transformed_target):
            #         raise ValueError(
            #             f"Dataset 错误：'allowed_indices' 的长度 "
            #             f"({len(allowed_indices) if allowed_indices else 'None'}) "
            #             f"与 'target_tokens' + EOS 的长度 ({len(transformed_target)}) 不匹配。"
            #         )
            #     target_allowed_indices_list.append(allowed_indices)
        # 填充标签序列，用-100忽略填充部分的损失
        # max_label_len = max(len(lbl) for lbl in labels_list)
        max_label_len = max(unpadded_label_lengths)
        for i in range(len(labels_list)):
            padding_len = max_label_len - unpadded_label_lengths[i]
            labels_list[i] = labels_list[i] + [-100] * padding_len
        # 添加 loss_mask，1代表可计算loss，0代表mask
        batch_size = len(features)
        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": (torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id).long(),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
        if has_vocab_mask_info:
            padded_loss_masks = []
            # 这里的 V (vocab_size) 是从 mask.shape[1] 自动推断的
            V = loss_mask_list[0].shape[1] 
            
            for i in range(len(loss_mask_list)):
                mask = loss_mask_list[i] # shape [seq_len, V]
                padding_len = max_label_len - mask.shape[0] # mask.shape[0] == unpadded_label_lengths[i]
                
                # 使用 F.pad 进行高效填充
                # (0, 0) -> 不填充最后一个维度 (V)
                # (0, padding_len) -> 在倒数第二个维度 (seq_len) 的 *末尾* 填充 padding_len 个 0
                padded_mask = torch.nn.functional.pad(
                    mask, (0, 0, 0, padding_len), "constant", 0.0
                )
                padded_loss_masks.append(padded_mask)
            
            # 4. 将列表堆叠成批次
            #    [B, max_label_len, V]
            batch["loss_mask"] = torch.stack(padded_loss_masks, dim=0)
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
            transformed_source = source_tokens
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