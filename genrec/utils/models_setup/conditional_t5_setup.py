from transformers import T5ForConditionalGeneration,T5Config


def create_t5_model(vocab_size: int, model_config: dict) -> T5ForConditionalGeneration:
    """
    创建标准的T5模型，根据提供的配置参数
    """
    config = T5Config(
        vocab_size=vocab_size,
        d_model = model_config['d_model'],  # 计算 d_model
        d_kv=model_config['d_kv'],
        d_ff=model_config['d_ff'],
        num_layers=model_config['num_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        num_heads=model_config['num_heads'],
        dropout_rate=model_config['dropout_rate'],
        tie_word_embeddings=model_config['tie_word_embeddings'],
        pad_token_id=0,  # 根据您的tokenizer设置
        eos_token_id=1,   # 根据您的tokenizer设置
        decoder_start_token_id=0,  # 通常与pad_token_id相同
    )
    
    model = T5ForConditionalGeneration(config)
    return model
