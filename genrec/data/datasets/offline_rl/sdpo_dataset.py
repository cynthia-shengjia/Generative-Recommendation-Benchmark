from typing import Dict, List, Any, Union
import random
from ..base_dataset import BaseSeqRecDataset

class SDPODataset(BaseSeqRecDataset):
    """SDPO (Offline RL) 数据集"""
    
    def __init__(self, *args, **kwargs):
        # 在初始化前设置负采样相关属性
        self.neg_num = 4
        super().__init__(*args, **kwargs)

    def _create_samples(self) -> List[Dict[str, Any]]:
        """创建样本（带负采样）"""
        samples = []
        max_item_seq_len = self.config['max_seq_len']
        
        for user_id, item_seq in self.user_seqs.items():
            if self.mode == 'train':
                # 训练集：截断最后两个物品
                item_seq = item_seq[:-2]
                for i in range(1, len(item_seq)):
                    history = item_seq[:i]
                    target = item_seq[i]
                    
                    if len(history) > max_item_seq_len:
                        history = history[-max_item_seq_len:]
                    
                    # 负采样
                    user_interacted = set(item_seq[:i+1])
                    candidate_negatives = [
                        item for item in self.all_items 
                        if item not in user_interacted
                    ]
                    negative_items = random.sample(
                        candidate_negatives, 
                        self.neg_num
                    )
                    
                    samples.append({
                        'user_id': user_id,
                        'history_items': history,
                        'target_item': target,
                        'negative_items': negative_items,
                    })
            
            elif self.mode == 'valid':
                # 验证集
                if len(item_seq) < 3:
                    continue
                history = item_seq[:-2]
                target = item_seq[-2]
                if len(history) > max_item_seq_len:
                    history = history[-max_item_seq_len:]
                
                # 负采样
                user_interacted = set(item_seq[:-1])
                candidate_negatives = [
                    item for item in self.all_items 
                    if item not in user_interacted
                ]
                negative_items = random.sample(
                    candidate_negatives, 
                    self.neg_num
                )
                
                samples.append({
                    'user_id': user_id,
                    'history_items': history,
                    'target_item': target,
                    'negative_items': negative_items,
                })
            
            elif self.mode == 'test':
                # 测试集
                if len(item_seq) < 2:
                    continue
                history = item_seq[:-1]
                target = item_seq[-1]
                if len(history) > max_item_seq_len:
                    history = history[-max_item_seq_len:]
                
                # 负采样
                user_interacted = set(item_seq)
                candidate_negatives = [
                    item for item in self.all_items 
                    if item not in user_interacted
                ]
                negative_items = random.sample(
                    candidate_negatives, 
                    self.neg_num
                )
                
                samples.append({
                    'user_id': user_id,
                    'history_items': history,
                    'target_item': target,
                    'negative_items': negative_items,
                })
        
        return samples
    
    def __getitem__(self, index: int) -> Dict[str, Union[int, List[int], Dict]]:
        sample = self.samples[index]
        history_items = sample['history_items']
        target_item = sample['target_item']
        negative_items = sample['negative_items']
        user_id = sample['user_id']
        
        # 历史物品转token
        source_tokens = []
        for item in history_items:
            source_tokens.extend(self._get_item_tokens(item))
        
        # 目标物品转token
        target_tokens = self._get_item_tokens(target_item)
        
        # 负样本转token
        rejected_tokens = {}
        for i, negative_item in enumerate(negative_items):
            rejected_tokens[i] = self._get_item_tokens(negative_item)
        
        return {
            'user_token': self.tokenizer.get_user_token(user_id),
            'source_tokens': source_tokens,
            'target_tokens': target_tokens,
            'rejected_tokens': rejected_tokens,
            'target_id': target_item,
        }