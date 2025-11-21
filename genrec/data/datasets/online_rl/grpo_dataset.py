from generative.tiger_dataset import TigerDataset

class GRPODataset(TigerDataset):
    """
    GRPO (Online RL) 数据集
    
    与 TigerDataset 完全相同，因为 GRPO 在训练时动态生成负样本，
    数据集本身不需要预先采样负样本
    """
    pass