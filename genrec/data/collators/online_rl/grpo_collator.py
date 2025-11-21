from generative.tiger_collator import TigerDataCollator

class GRPODataCollator(TigerDataCollator):
    """
    GRPO (Online RL) DataCollator
    
    与 TigerDataCollator 完全相同，因为 GRPO 的负样本处理在训练循环中完成
    """
    pass