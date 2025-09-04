from transformers import T5Config

class CustomT5Config(T5Config):
    def __init__(
        self,
        vocab_size: int = 1,
        d_kv: int = 64,               # 每个注意力头的维度
        num_heads: int = 6,           # 注意力头数
        d_ff: int = 1024,             # 前馈网络维度
        num_layers: int = 4,          # 编码器层数
        num_decoder_layers: int = 4,  # 解码器层数
        dropout_rate: float = 0.1,    # dropout率
        **kwargs
    ):
        # 计算 d_model = d_kv * num_heads
        d_model = d_kv * num_heads
        
        # 设置默认参数以匹配配置
        defaults = {
            'd_model': d_model,           # 模型维度 = d_kv * num_heads
            'vocab_size': vocab_size,
            'd_kv': d_kv,                 # 每个注意力头的维度
            'd_ff': d_ff,                 # 前馈网络维度
            'num_layers': num_layers,     # 编码器层数
            'num_decoder_layers': num_decoder_layers,  # 解码器层数
            'num_heads': num_heads,       # 注意力头数
            'dropout_rate': dropout_rate, # dropout率
            'bos_token_id': 0
        }
        
        # 更新默认值，但允许通过kwargs覆盖
        defaults.update(kwargs)
        
        super().__init__(**defaults)
        self.vocab_size = vocab_size
        # latent_dim 现在等于 d_model
        self.latent_dim = d_model
        
    def to_dict(self):
        output = super().to_dict()
        output["vocab_size"] = self.vocab_size
        output["latent_dim"] = self.latent_dim
        return output