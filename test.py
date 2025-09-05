import os
import torch
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import logging 
from model_train import create_custom_t5_model, train_model

# 假设 model_train 模块中有 create_custom_t5_model 函数
# from model_train import create_custom_t5_model, train_model

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

# 创建模型
model = create_custom_t5_model(
    vocab_size=1000, d_kv=model_config['d_kv'], d_ff=model_config['d_ff'],
    num_layers=model_config['num_layers'], num_decoder_layers=model_config['num_decoder_layers'],
    num_heads=model_config['num_heads'], dropout_rate=model_config['dropout_rate'],
    tie_word_embeddings=model_config['tie_word_embeddings']
)

# 准备输入
encoder_input_ids = torch.tensor([
    [101, 102, 103, 104, 0, 0],  # 第一个用户序列
    [201, 202, 203, 204, 205, 206]  # 第二个用户序列
])

encoder_attention_mask = torch.tensor([
    [1, 1, 1, 1, 0, 0],  # 第一个用户的掩码
    [1, 1, 1, 1, 1, 1]   # 第二个用户的掩码
])

max_length = 5
batch_size = 2
num_beams = 5

# 使用beam search生成并获取概率分数
outputs = model.generate(
    encoder_input_ids=encoder_input_ids,
    encoder_attention_mask=encoder_attention_mask,
    max_length=max_length,
    num_beams=num_beams,
    num_return_sequences=num_beams,
    early_stopping=True,
    decoder_start_token_id=0,
    output_scores=True,  # 获取分数
    return_dict_in_generate=True,  # 返回字典格式,
)

# 获取生成的序列和分数
generated_ids = outputs.sequences
sequences_scores = outputs.sequences_scores  # 对数概率



# 将分数转换为概率
probabilities = torch.exp(sequences_scores)

# 重塑结果以便按用户分组
generated_ids_reshaped = generated_ids.view(batch_size, num_beams, -1)[:, :, 1:]  # 跳过第一个token
probabilities_reshaped = probabilities.view(batch_size, num_beams)

print("原始生成结果:")
print(generated_ids_reshaped)
print("\n每个序列的概率:")
print(probabilities_reshaped)

# 对每个用户的生成序列按概率排序
sorted_results = []
for i in range(batch_size):
    # 获取当前用户的所有序列和概率
    user_sequences = generated_ids_reshaped[i]
    user_probs = probabilities_reshaped[i]
    
    # 按概率从高到低排序
    sorted_indices = torch.argsort(user_probs, descending=True)
    sorted_sequences = user_sequences[sorted_indices]
    sorted_probs = user_probs[sorted_indices]
    
    # 添加到结果列表
    sorted_results.append({
        'user_id': i,
        'sequences': sorted_sequences,
        'probabilities': sorted_probs
    })


# # 打印排序后的结果
# print("\n按概率排序后的结果:")
# for user_result in sorted_results:
#     print(f"\n用户 {user_result['user_id'] + 1} 的生成序列 (按概率从高到低):")
#     for j, (seq, prob) in enumerate(zip(user_result['sequences'], user_result['probabilities'])):
#         # 移除填充token (假设pad_token_id=0)
#         non_pad_tokens = [token.item() for token in seq if token != 0]
#         print(f"  排名 {j+1}: {non_pad_tokens}, 概率: {prob.item():.6f}")

# # 如果您需要将结果保存到文件
# def save_sorted_results(sorted_results, filename):
#     """将排序后的结果保存到JSON文件"""
#     results_dict = {}
#     for user_result in sorted_results:
#         user_id = user_result['user_id']
#         results_dict[user_id] = {
#             'sequences': user_result['sequences'].tolist(),
#             'probabilities': user_result['probabilities'].tolist()
#         }
    
#     with open(filename, 'w') as f:
#         json.dump(results_dict, f, indent=2)
    
#     print(f"\n结果已保存到 {filename}")

# # 保存结果到文件
# save_sorted_results(sorted_results, "sorted_generation_results.json")