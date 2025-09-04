from transformers import T5Config

class CustomT5Config(T5Config):
    def __init__(
        self,
        codebook_size: int,
        latent_dim: int = 128,  # 根据配置设置为128
        **kwargs
    ):
        # 设置默认参数以匹配配置
        defaults = {
            'd_model': 128,           # 模型维度
            'd_kv': 64,               # 每个注意力头的维度
            'd_ff': 1024,             # 前馈网络维度
            'num_layers': 4,          # 编码器层数
            'num_decoder_layers': 4,  # 解码器层数
            'num_heads': 6,           # 注意力头数
            'dropout_rate': 0.1,      # dropout率
        }
        
        # 更新默认值，但允许通过kwargs覆盖
        defaults.update(kwargs)
        
        super().__init__(**defaults)
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        
    def to_dict(self):
        output = super().to_dict()
        output["codebook_size"] = self.codebook_size
        output["latent_dim"] = self.latent_dim
        return output