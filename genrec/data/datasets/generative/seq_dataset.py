from typing import Dict, List, Any, Union
from ..base_dataset import BaseSeqRecDataset

class SeqDataset(BaseSeqRecDataset):
    
    def _create_samples(self) -> List[Dict[str, Any]]:
        samples = []
        max_item_seq_len = self.config['max_seq_len']
        
        for user_id, item_seq in self.user_seqs.items():
            if self.mode == 'train':
                # 先截断max_item_seq_len 长度，再滑窗
                item_seq = item_seq[-(max_item_seq_len+2):-2]
                for i in range(1, len(item_seq)):
                    history = item_seq[:i]
                    target = item_seq[i]
                    # if len(history) > max_item_seq_len:
                    #     history = history[-max_item_seq_len:]
                    samples.append({
                        'user_id': user_id,
                        'history_items': history,
                        'target_item': target
                    })
            
            elif self.mode == 'valid':
                # 验证集：倒数第二个物品作为目标
                if len(item_seq) < 3:
                    continue
                history = item_seq[:-2]
                target = item_seq[-2]
                if len(history) > max_item_seq_len:
                    history = history[-max_item_seq_len:]
                samples.append({
                    'user_id': user_id,
                    'history_items': history,
                    'target_item': target
                })
            
            elif self.mode == 'test':
                # 测试集：最后一个物品作为目标
                if len(item_seq) < 2:
                    continue
                history = item_seq[:-1]
                target = item_seq[-1]
                if len(history) > max_item_seq_len:
                    history = history[-max_item_seq_len:]
                samples.append({
                    'user_id': user_id,
                    'history_items': history,
                    'target_item': target
                })
        
        return samples
    
    def __getitem__(self, index: int) -> Dict[str, Union[int, List[int]]]:
        sample = self.samples[index]
        history_items = sample['history_items']
        target_item = sample['target_item']
        user_id = sample['user_id']
        
        # 将历史物品转换为token序列
        source_tokens = []
        for item in history_items:
            source_tokens.extend(self._get_item_tokens(item))
        
        # 将目标物品转换为token序列
        target_tokens = self._get_item_tokens(target_item)
        
        return {
            'user_token': self.tokenizer.get_user_token(user_id),
            'source_tokens': source_tokens,
            'target_tokens': target_tokens,
            'target_id': target_item,
        }