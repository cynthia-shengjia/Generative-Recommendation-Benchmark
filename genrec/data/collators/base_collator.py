from dataclasses import dataclass
from typing import Dict, List, Any
import torch

@dataclass
class BaseSeqRecDataCollator:
    """序列推荐 DataCollator 基类"""
    max_seq_len: int
    pad_token_id: int
    eos_token_id: int
    tokens_per_item: int = 4
    label_pad_token_id: int = -100
    
    def process_encoder_input(self, source_tokens: List[int]) -> Dict[str, Any]:
        """
        处理 encoder 输入（历史序列）- 左 padding
        
        Args:
            source_tokens: 历史物品的 token 序列
            
        Returns:
            包含 input_ids 的字典
        """
        transformed_source = source_tokens
        
        # 截断或左 padding
        if len(transformed_source) > self.max_seq_len:
            transformed_source = transformed_source[-self.max_seq_len:]
        else:
            padding_length = self.max_seq_len - len(transformed_source)
            transformed_source = [self.pad_token_id] * padding_length + transformed_source
        
        return {"input_ids": transformed_source}
    
    def process_decoder_target(self, target_tokens: List[int]) -> Dict[str, Any]:
        """
        处理 decoder 目标序列 - 添加 EOS
        
        Args:
            target_tokens: 目标物品的 token 序列
            
        Returns:
            包含 labels 和 unpadded_length 的字典
        """
        transformed_target = list(target_tokens)
        transformed_target.append(self.eos_token_id)
        
        return {
            "labels": transformed_target,
            "unpadded_length": len(transformed_target),
        }
    
    def pad_labels(self, labels_list: List[List[int]], max_label_len: int) -> List[List[int]]:
        """
        右 padding labels
        
        Args:
            labels_list: labels 列表
            max_label_len: 最大 label 长度
            
        Returns:
            padding 后的 labels 列表
        """
        padded_labels = []
        for labels in labels_list:
            padding_len = max_label_len - len(labels)
            padded_labels.append(labels + [self.label_pad_token_id] * padding_len)
        return padded_labels
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将 features 转换为 batch - 子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement __call__()")