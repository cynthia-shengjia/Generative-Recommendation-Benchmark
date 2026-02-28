from dataclasses import dataclass
from typing import Dict, List, Any
import torch

@dataclass
class BaseSeqRecDataCollator:

    max_seq_len: int
    pad_token_id: int
    eos_token_id: int
    tokens_per_item: int = 4
    label_pad_token_id: int = -100
    
    def process_encoder_input(self, source_tokens: List[int]) -> Dict[str, Any]:

        transformed_source = source_tokens
        
        if len(transformed_source) > self.max_seq_len:
            transformed_source = transformed_source[-self.max_seq_len:]
        else:
            padding_length = self.max_seq_len - len(transformed_source)
            transformed_source = [self.pad_token_id] * padding_length + transformed_source
        
        return {"input_ids": transformed_source}
    
    def process_decoder_target(self, target_tokens: List[int]) -> Dict[str, Any]:
        """
        Add EOS
        """
        transformed_target = list(target_tokens)
        transformed_target.append(self.eos_token_id)
        
        return {
            "labels": transformed_target,
            "unpadded_length": len(transformed_target),
        }
    
    def pad_labels(self, labels_list: List[List[int]], max_label_len: int) -> List[List[int]]:
        """
        right padding labels
        """
        padded_labels = []
        for labels in labels_list:
            padding_len = max_label_len - len(labels)
            padded_labels.append(labels + [self.label_pad_token_id] * padding_len)
        return padded_labels
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement __call__()")