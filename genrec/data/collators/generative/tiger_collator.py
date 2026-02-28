from dataclasses import dataclass  
from typing import Dict, List, Any  
import torch  
from ..base_collator import BaseSeqRecDataCollator  
  
  
@dataclass  
class TigerDataCollator(BaseSeqRecDataCollator):  
    """Tiger 生成式推荐 DataCollator"""  
    max_seq_len: int  
    pad_token_id: int  
    eos_token_id: int  
    mode: str = "train"  # 'train', 'valid', 'test'
      
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:  
        """  
        input features: List of  
            {  
                'source_tokens': [...],  
                'target_tokens': [...],  
                'user_token': int,  
                'target_id': int,
            }  
          
        output batch:  
            {  
                'input_ids': [B, max_seq_len],  
                'attention_mask': [B, max_seq_len],  
                'labels': [B, max_label_len],  # train/valid model
                'label_id': List[int],  # test model
            }  
        """  
        input_ids_list = []  
        for feature in features:  
            source_tokens = feature["source_tokens"]
            result = self.process_encoder_input(source_tokens)
            transformed_source = result["input_ids"]
            input_ids_list.append(transformed_source)
        

        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": (
                torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id
            ).long(),
        }
        
        if self.mode in ["train", "valid"]:
            labels_list = []  
            unpadded_lengths = []  
              
            for feature in features:  
                target_tokens = feature["target_tokens"]
                
                result = self.process_decoder_target(target_tokens)
                transformed_target = result["labels"]
                
                labels_list.append(transformed_target)  
                unpadded_lengths.append(len(transformed_target))
              
            max_label_len = max(unpadded_lengths)  
            for i in range(len(labels_list)):
                padding_len = max_label_len - unpadded_lengths[i]
                labels_list[i] = labels_list[i] + [-100] * padding_len
              
            batch["labels"] = torch.tensor(labels_list, dtype=torch.long)
        
        label_ids = [feature["target_id"] for feature in features]
        batch["label_id"] = torch.tensor(label_ids, dtype=torch.long)
          
        return batch