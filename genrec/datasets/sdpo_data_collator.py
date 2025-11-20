from dataclasses import dataclass
from typing import Dict, List, Any
import torch

@dataclass
class S_DPOSeqRecDataCollator:
    """S-DPO 专用的序列推荐 DataCollator（Encoder-Decoder 架构）"""
    max_seq_len: int          # 历史序列的最大长度
    pad_token_id: int         # padding token
    eos_token_id: int         # EOS token
    tokens_per_item: int = 4  # 每个物品的 token 数量
    label_pad_token_id: int = -100  # labels 的 padding token

    def process_encoder_input(self, source_tokens: List[int]) -> Dict[str, Any]:
        """
        处理 encoder 输入（历史序列）
        
        返回:
            input_ids: 左 padding 的历史序列
        """
        transformed_source = source_tokens
        
        # 截断或左 padding 源序列
        if len(transformed_source) > self.max_seq_len:
            transformed_source = transformed_source[-self.max_seq_len:]  # 保留最后的
        else:
            padding_length = self.max_seq_len - len(transformed_source)
            transformed_source = [self.pad_token_id] * padding_length + transformed_source  # 左 padding
        
        return {
            "input_ids": transformed_source,
        }
    
    def process_decoder_target(self, target_tokens: List[int]) -> Dict[str, Any]:
        """
        处理 decoder 目标序列
        
        返回:
            labels: 目标序列 + EOS（未 padding）
            unpadded_length: 未 padding 的长度
        """
        transformed_target = list(target_tokens)
        transformed_target.append(self.eos_token_id)  # 添加 EOS
        
        return {
            "labels": transformed_target,
            "unpadded_length": len(transformed_target),
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将 features 转换为 S-DPO 需要的 batch 格式（Encoder-Decoder）
        
        输入 features: List of
            {
                'source_tokens': [...],
                'chosen_tokens': [...],
                'rejected_tokens': {1: [...], 2: [...], 3: [...]},
            }
        
        输出 batch:
            {
                'input_ids': [B, max_seq_len],              # Encoder 输入（共享）
                'attention_mask': [B, max_seq_len],         # Encoder attention mask
                'chosen_labels': [B, max_label_len],        # Decoder 目标（chosen）
                'rejected_labels': [B, N, max_label_len],   # Decoder 目标（rejected）
            }
        """
        batch_size = len(features)
        
        # 找出最大负样本数量
        max_neg_num = max(
            len(feature.get("rejected_tokens", {})) for feature in features
        )
        
        # ===== 处理 Encoder 输入（共享） =====
        input_ids_list = []
        
        for feature in features:
            result = self.process_encoder_input(feature["source_tokens"])
            input_ids_list.append(result["input_ids"])
        
        # ===== 处理 Chosen Labels =====
        chosen_labels_list = []
        chosen_unpadded_lengths = []
        
        for feature in features:
            result = self.process_decoder_target(feature["target_tokens"])
            chosen_labels_list.append(result["labels"])
            chosen_unpadded_lengths.append(result["unpadded_length"])
        
        # ===== 处理 Rejected Labels (多个) =====
        rejected_labels_batch = []
        rejected_unpadded_lengths_batch = []
        
        for feature in features:
            sample_rejected_labels = []
            sample_rejected_unpadded_lengths = []
            
            # 直接遍历 rejected_tokens 的 values
            rejected_tokens_dict = feature.get("rejected_tokens", {})
            for rejected_tokens in rejected_tokens_dict.values():
                result = self.process_decoder_target(rejected_tokens)
                sample_rejected_labels.append(result["labels"])
                sample_rejected_unpadded_lengths.append(result["unpadded_length"])
            
            # 如果这个样本的 rejected 数量少于 max_neg_num，用空的补齐
            num_rejected = len(rejected_tokens_dict)
            for _ in range(max_neg_num - num_rejected):
                result = self.process_decoder_target([])  # 空的 rejected
                sample_rejected_labels.append(result["labels"])
                sample_rejected_unpadded_lengths.append(result["unpadded_length"])
            
            rejected_labels_batch.append(sample_rejected_labels)
            rejected_unpadded_lengths_batch.append(sample_rejected_unpadded_lengths)
        
        # ===== Padding labels (右 padding) =====
        # 找出 chosen 和 rejected 中最长的 label
        max_label_len = max(chosen_unpadded_lengths)
        for sample_lengths in rejected_unpadded_lengths_batch:
            max_label_len = max(max_label_len, max(sample_lengths))
        
        # Pad chosen labels (右 padding)
        for i in range(batch_size):
            padding_len = max_label_len - chosen_unpadded_lengths[i]
            chosen_labels_list[i] = chosen_labels_list[i] + [self.label_pad_token_id] * padding_len
        
        # Pad rejected labels (右 padding)
        for i in range(batch_size):
            for j in range(max_neg_num):
                padding_len = max_label_len - rejected_unpadded_lengths_batch[i][j]
                rejected_labels_batch[i][j] = rejected_labels_batch[i][j] + [self.label_pad_token_id] * padding_len
        
        # ===== 转换为 tensor =====
        batch = {
            # Encoder 输入（共享）
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),  # [B, max_seq_len]
            "attention_mask": (
                torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id
            ).long(),  # [B, max_seq_len]
            
            # Decoder 目标
            "chosen_labels": torch.tensor(chosen_labels_list, dtype=torch.long),  # [B, max_label_len]
            "rejected_labels": torch.tensor(rejected_labels_batch, dtype=torch.long),  # [B, N, max_label_len]
        }
        
        return batch