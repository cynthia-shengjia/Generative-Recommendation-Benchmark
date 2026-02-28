from dataclasses import dataclass
from typing import Dict, List, Any
import torch
from ..base_collator import BaseSeqRecDataCollator



@dataclass
class SDPODataCollator(BaseSeqRecDataCollator):
    """SDPO (Offline RL) DataCollator"""
    max_seq_len: int  = 100
    pad_token_id: int  = 1
    eos_token_id: int  = 1
    mode: str = "train"  # 'train', 'valid', 'test'
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        
        input features: List of
            {
                'source_tokens': [...],
                'target_tokens': [...],  # chosen
                'rejected_tokens': {0: [...], 1: [...], ...},
                'user_token': int,
                'target_id': int,
            }
        
        output batch:
            {
                'input_ids': [B, max_seq_len],
                'attention_mask': [B, max_seq_len],
                'chosen_labels': [B, max_label_len],
                'rejected_labels': [B, N, max_label_len],
            }
        """
        batch_size = len(features)
        
        max_neg_num = max(
            len(feature.get("rejected_tokens", {})) for feature in features
        )
        
        # Encoder input (share)
        input_ids_list = []
        for feature in features:
            result = self.process_encoder_input(feature["source_tokens"])
            input_ids_list.append(result["input_ids"])
        
        # Chosen Labels
        chosen_labels_list = []
        chosen_unpadded_lengths = []
        
        for feature in features:
            result = self.process_decoder_target(feature["target_tokens"])
            chosen_labels_list.append(result["labels"])
            chosen_unpadded_lengths.append(result["unpadded_length"])
        
        # Rejected Labels
        rejected_labels_batch = []
        rejected_unpadded_lengths_batch = []
        
        for feature in features:
            sample_rejected_labels = []
            sample_rejected_unpadded_lengths = []
            
            rejected_tokens_dict = feature.get("rejected_tokens", {})
            for rejected_tokens in rejected_tokens_dict.values():
                result = self.process_decoder_target(rejected_tokens)
                sample_rejected_labels.append(result["labels"])
                sample_rejected_unpadded_lengths.append(result["unpadded_length"])
            
            # max_neg_num
            num_rejected = len(rejected_tokens_dict)
            for _ in range(max_neg_num - num_rejected):
                result = self.process_decoder_target([])
                sample_rejected_labels.append(result["labels"])
                sample_rejected_unpadded_lengths.append(result["unpadded_length"])
            
            rejected_labels_batch.append(sample_rejected_labels)
            rejected_unpadded_lengths_batch.append(sample_rejected_unpadded_lengths)
        
        # find max label length
        max_label_len = max(chosen_unpadded_lengths)
        for sample_lengths in rejected_unpadded_lengths_batch:
            if sample_lengths:
                max_label_len = max(max_label_len, max(sample_lengths))
        
        # Padding chosen labels
        chosen_labels_list = self.pad_labels(chosen_labels_list, max_label_len)
        
        # Padding rejected labels
        for i in range(batch_size):
            for j in range(max_neg_num):
                padding_len = max_label_len - rejected_unpadded_lengths_batch[i][j]
                rejected_labels_batch[i][j] = (
                    rejected_labels_batch[i][j] + 
                    [self.label_pad_token_id] * padding_len
                )
        

        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": (
                torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id
            ).long(),
            "chosen_labels": torch.tensor(chosen_labels_list, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels_batch, dtype=torch.long),
        }
        

        if self.mode in ["test", "valid"]:
            label_ids = [feature["target_id"] for feature in features]
            batch["label_id"] = label_ids

        return batch