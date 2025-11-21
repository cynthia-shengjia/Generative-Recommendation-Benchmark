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
        将 features 转换为 batch  
          
        输入 features: List of  
            {  
                'source_tokens': [...],  
                'target_tokens': [...],  
                'user_token': int,  
                'target_id': int,
            }  
          
        输出 batch:  
            {  
                'input_ids': [B, max_seq_len],  
                'attention_mask': [B, max_seq_len],  
                'labels': [B, max_label_len],  # train/valid 模式
                'label_id': List[int],  # test 模式（valid 模式也可能需要）
            }  
        """  
          
        # 处理 Encoder 输入  
        input_ids_list = []  
        for feature in features:  
            source_tokens = feature["source_tokens"]
            
            # 应用偏移到源序列（如果 process_encoder_input 有偏移逻辑）
            result = self.process_encoder_input(source_tokens)
            transformed_source = result["input_ids"]
            
            # 截断或填充源序列
            if len(transformed_source) > self.max_seq_len:
                transformed_source = transformed_source[-self.max_seq_len:]
            else:
                padding_length = self.max_seq_len - len(transformed_source)
                transformed_source = [self.pad_token_id] * padding_length + transformed_source
            
            input_ids_list.append(transformed_source)
        
        # 构建基础 batch
        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": (
                torch.tensor(input_ids_list, dtype=torch.long) != self.pad_token_id
            ).long(),
        }
        
        # 训练和验证模式需要 labels
        if self.mode in ["train", "valid"]:
            labels_list = []  
            unpadded_lengths = []  
              
            for feature in features:  
                target_tokens = feature["target_tokens"]
                
                # 应用偏移到目标序列
                result = self.process_decoder_target(target_tokens)
                transformed_target = result["labels"]
                
                # 添加 EOS token
                transformed_target.append(self.eos_token_id)
                
                labels_list.append(transformed_target)  
                unpadded_lengths.append(len(transformed_target))
              
            # Padding labels，用 -100 忽略填充部分的损失
            max_label_len = max(unpadded_lengths)  
            for i in range(len(labels_list)):
                padding_len = max_label_len - unpadded_lengths[i]
                labels_list[i] = labels_list[i] + [-100] * padding_len
              
            batch["labels"] = torch.tensor(labels_list, dtype=torch.long)
        
        # 测试模式（以及可能的验证模式）需要 label_id 用于评估
        if self.mode in ["test", "valid"]:
            label_ids = [feature["target_id"] for feature in features]
            batch["label_id"] = label_ids
          
        return batch