# genrec/rewards/base_reward.py

from abc import ABC, abstractmethod
from typing import List

class BaseReward(ABC):
    """
    Base class for reward functions in online RL training.
    """
    
    @abstractmethod
    def __call__(
        self, 
        generated_items: List[int], 
        target_items: List[int], 
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for generated items.
        
        Args:
            generated_items: List of generated item IDs
            target_items: List of target item IDs
            **kwargs: Additional arguments (e.g., num_generations, ranks)
        
        Returns:
            List of rewards
        """
        raise NotImplementedError("Subclasses must implement __call__")