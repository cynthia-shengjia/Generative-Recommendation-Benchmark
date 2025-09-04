
import os
import torch
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import logging 
from model_train import create_custom_t5_model, train_model
model_config = {
        'dataset_name': "hello",
        'data_interaction_files': './data/Beauty/user2item.pkl',
        'data_text_files': './data/Beauty/item2title.pkl',
        'max_seq_len': 20, 'padding_side': 'left', 'ignored_label': -100,
        'vocab_size': 256 * 4, 'd_kv': 64, 'd_ff': 1024, 'num_layers': 4,
        'num_decoder_layers': 4, 'num_heads': 6, 'dropout_rate': 0.1,
        'tie_word_embeddings': True, 'batch_size': 128, 'learning_rate': 0.001,
        'num_epochs': 2, 'num_steps': None,
        'model_save_path': "asd",
        'checkpoint_dir': "asd", 'device': "cpu",
    }



# # 创建模型
model = create_custom_t5_model(
            vocab_size=1000, d_kv=model_config['d_kv'], d_ff=model_config['d_ff'],
            num_layers=model_config['num_layers'], num_decoder_layers=model_config['num_decoder_layers'],
            num_heads=model_config['num_heads'], dropout_rate=model_config['dropout_rate'],
            tie_word_embeddings=model_config['tie_word_embeddings']
        )
# 准备输入
encoder_input_ids = torch.tensor([[101, 102, 103, 104]])  # 示例输入
encoder_attention_mask = torch.ones_like(encoder_input_ids)

# 使用beam search生成
generated_ids = model.generate(
    encoder_input_ids=encoder_input_ids,
    encoder_attention_mask=encoder_attention_mask,
    max_length=20,
    num_beams=5,
    early_stopping=True,
    decoder_start_token_id=0,  # 对于T5，通常是pad_token_id
)

print("Generated IDs:", generated_ids)