from transformers import T5ForConditionalGeneration,T5Config
from genrec.models.LETTER.LETTERT5 import LETTERT5ForConditionalGeneration

def create_letter_model(vocab_size: int, model_config: dict) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=vocab_size,
        d_model = model_config['d_model'], 
        d_kv=model_config['d_kv'],
        d_ff=model_config['d_ff'],
        num_layers=model_config['num_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        num_heads=model_config['num_heads'],
        dropout_rate=model_config['dropout_rate'],
        tie_word_embeddings=model_config['tie_word_embeddings'],
        pad_token_id=0,
        eos_token_id=1, 
        decoder_start_token_id=0, 
        tau=model_config['tau'],
    )
    
    model = LETTERT5ForConditionalGeneration(config)
    return model
