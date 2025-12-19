# genrec/rewards/grpo_reward.py

import math
from typing import List
from .base_reward import BaseReward

class GRPOReward(BaseReward):
    """
    GRPO reward with optional NDCG-based ranking penalty.
    
    Combines binary match reward with NDCG-based penalties for negative samples.
    """
    
    def __init__(self, use_ndcg: bool = True, ndcg_weight: float = 0.5):
        """
        Initialize GRPOReward.
        
        Args:
            use_ndcg: Whether to use NDCG-based ranking penalty
            ndcg_weight: Weight for NDCG penalty (0-1)
        """
        self.use_ndcg = use_ndcg
        self.ndcg_weight = ndcg_weight
    
    def __call__(
        self, 
        generated_items: List[int], 
        target_items: List[int], 
        num_generations: int = None,
        **kwargs
    ) -> List[float]:
        """
        Compute GRPO rewards.
        
        Args:
            generated_items: List of generated item IDs [B * num_generations]
            target_items: List of target item IDs [B * num_generations]
            num_generations: Number of generations per sample (required)
            **kwargs: Unused
        
        Returns:
            List of rewards
        """
        if num_generations is None:
            raise ValueError("num_generations is required for GRPOReward")
        
        # Precompute NDCG penalties
        ndcg_penalties = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
        ndcg_sum = sum(ndcg_penalties)
        ndcg_penalties = [-elm / ndcg_sum for elm in ndcg_penalties]
        
        rewards = []
        
        # Process by groups
        for group_idx in range(len(generated_items) // num_generations):
            start_idx = group_idx * num_generations
            end_idx = start_idx + num_generations
            
            group_gen_items = generated_items[start_idx:end_idx]
            group_target_items = target_items[start_idx:end_idx]
            
            # Note: group_gen_items are already sorted by probability (high to low)
            for rank, (gen_item, target_item) in enumerate(zip(group_gen_items, group_target_items)):
                # Base match reward
                match_reward = 1.0 if gen_item == target_item else 0.0
                
                if not self.use_ndcg:
                    final_reward = match_reward
                else:
                    if match_reward == 1.0:  # Positive sample
                        # Positive samples get 0 NDCG reward
                        final_reward = (1 - self.ndcg_weight) * match_reward + self.ndcg_weight * 0.0
                    else:  # Negative sample
                        # Negative samples get ranking-based penalty
                        final_reward = (1 - self.ndcg_weight) * match_reward + self.ndcg_weight * ndcg_penalties[rank]
                
                rewards.append(final_reward)
        
        return rewards