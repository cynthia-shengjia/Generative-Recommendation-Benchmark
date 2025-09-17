from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
from torch.nn.utils.rnn import pad_sequence
from genrec.cbs_structure.MSL_Tire import Trie

@dataclass
class TrainSeqRecDataCollator:
    max_seq_len: int
    pad_token_id: int
    eos_token_id: int
    tokens_per_item: int = 4
    trie: Optional[Trie] = None  # 添加Trie实例
    vocab_size: Optional[int] = None  # 添加词汇表大小
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        constrain_mask_list = []  # 用于存储constrain_mask

        for feature in features:
            source_tokens = feature["source_tokens"]
            target_tokens = feature["target_tokens"]
            user_token = feature['user_token']
            
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
            
            # 生成constrain_mask
            if self.trie and self.vocab_size is not None:
                constrain_mask = self._generate_constrain_mask(target_tokens)
                constrain_mask_list.append(constrain_mask)

        # 填充标签序列，用-100忽略填充部分的损失
        max_label_len = max(len(lbl) for lbl in labels_list)
        for i in range(len(labels_list)):
            labels_list[i] = labels_list[i] + [-100] * (max_label_len - len(labels_list[i]))


        # 处理constrain_mask的填充
        if constrain_mask_list:
            padded_constrain_mask = pad_sequence(
                constrain_mask_list, 
                batch_first=True, 
                padding_value=self.pad_token_id, # 填充值为0（无效）
                padding_side="right"
            )
 
        else:
            padded_constrain_mask = None
            

        result = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": (torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id).long(),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
        
        if padded_constrain_mask is not None:
            result["constrain_mask"] = padded_constrain_mask

        return result
    
    def _generate_constrain_mask(self, target_tokens: List[int]) -> torch.Tensor:
        """为目标token序列生成constrain_mask"""
        # 添加EOS token到目标序列

        target_with_eos = [0] + list(target_tokens)
        
        
        # 使用Trie获取每个位置的有效token
        allowed_tokens_list = self.trie.valid_tokens(target_with_eos)

        # 注意：valid_tokens返回的列表长度比输入序列多1
        # 第一个元素是空前缀的有效token，最后一个元素是完整序列后的有效token
        # 我们需要的是每个位置生成下一个token时的有效token，所以取前len(target_with_eos)个
        response_length = len(target_with_eos)
        
        # 初始化constrain_mask，形状为(response_length, vocab_size)
        constrain_mask = torch.zeros((response_length, self.vocab_size), dtype=torch.bool)
        
        # 填充constrain_mask
        for i in range(response_length):
            if i < len(allowed_tokens_list) - 1:
                allowed_tokens = allowed_tokens_list[i+1]  # 注意索引
                constrain_mask[i, allowed_tokens] = True

        return constrain_mask




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