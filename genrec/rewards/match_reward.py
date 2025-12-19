# genrec/rewards/match_reward.py

from typing import List
from .base_reward import BaseReward

class MatchReward(BaseReward):
    """
    Simple binary match reward.
    
    Returns 1.0 if generated item matches target, 0.0 otherwise.
    """
    
    def __init__(self):
        """Initialize MatchReward."""
        pass
    
    def __call__(
        self, 
        generated_items: List[int], 
        target_items: List[int], 
        **kwargs
    ) -> List[float]:
        """
        Compute binary match rewards.
        
        Args:
            generated_items: List of generated item IDs
            target_items: List of target item IDs
            **kwargs: Unused
        
        Returns:
            List of rewards (1.0 for match, 0.0 for mismatch)
        """
        rewards = []
        for gen_item, target_item in zip(generated_items, target_items):
            rewards.append(1.0 if gen_item == target_item else 0.0)
        return rewards