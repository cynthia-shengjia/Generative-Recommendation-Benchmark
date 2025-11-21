import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from genrec.tokenizers.TigerTokenizer import TigerTokenizer
from tools.utils import calc_ndcg, tokens_to_item_id
from typing import Optional, Dict, Any, List, Union, Callable
import math
import logging
import re
from transformers import T5ForConditionalGeneration,T5Config,Trainer
import numpy as np
import math
from transformers import EvalPrediction
from functools import partial
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import TrainerCallback, TrainingArguments, TrainerState
from genrec.trainers.generative.tiger_trainer import GenerativeTrainer
from genrec.utils.nni_utils import get_nni_params, update_config_with_nni, report_nni_metrics
from genrec.generation.trie import Trie,prefix_allowed_tokens_fn
import matplotlib.pyplot as plt
import seaborn as sns

def plot_encoder_attention_heatmap(
    attention_weights,
    input_ids,
    file_path="encoder_self_attention_heatmap.png",
    head_index=None, # 新增参数，用于在标题中注明是哪个头
    highlight_diagonal=True,
    draw_grid_lines=True
):
    """
    绘制并保存单个Encoder自注意力热力图。
    """
    # ... (这部分代码与之前完全相同，用于准备数据)
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().tolist()
    non_pad_tokens_indices = [i for i, token_id in enumerate(input_ids) if token_id != 0]
    if not non_pad_tokens_indices:
        print(f"警告：样本只包含padding tokens，跳过绘图。文件路径: {file_path}")
        return
    attention_weights = attention_weights[non_pad_tokens_indices, :][:, non_pad_tokens_indices]
    labels = [str(input_ids[i]) for i in non_pad_tokens_indices]
    seq_len = len(labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        attention_weights, xticklabels=labels, yticklabels=labels,
        cmap="viridis", annot=False, ax=ax
    )

    # --- 修改点：动态生成标题 ---
    title = "Encoder Self-Attention Heatmap"
    if head_index is not None:
        title += f" (Head {head_index})"
    else:
        title += " (Averaged Heads)"
    ax.set_title(title)
    # --- 结束修改点 ---

    if highlight_diagonal:
        ax.plot([0, seq_len], [0, seq_len], color='red', linestyle='--', linewidth=1.5)

    if draw_grid_lines and seq_len > 1:
        line_positions = [1] + [i + 1 for i in range(1, seq_len) if (i - 1) % 4 == 3]
        for pos in line_positions:
            if pos < seq_len:
                ax.axvline(x=pos, color='white', linestyle=':', linewidth=0.8)
                ax.axhline(y=pos, color='white', linestyle=':', linewidth=0.8)

    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
    plt.close(fig)
import numpy as np
import os
def calculate_item_attention_v2(attention_matrix, seq_len):
    """
    计算并比较 item 内部注意力和 item 间注意力的平均值 (版本2)。

    规则:
    - 忽略序列的第一个 token (index 0)。
    - 从第二个 token 开始，每 4 个 token 构成一个 item。
    - Intra-item (内部) 注意力: 
        - 版本1 (包含自身): 一个 item 内的 token 到该 item 内所有 token (包括自身) 的注意力。
        - 版本2 (不含自身): 一个 item 内的 token 到该 item 内其他 token 的注意力。
    - Inter-item (之间) 注意力: 一个 item 内的 token 到所有其他 items 内所有 token 的注意力 (双向)。

    参数:
    - attention_matrix (np.array): (seq_len, seq_len) 的注意力权重矩阵。
    - seq_len (int): 序列的有效长度 (不包含 padding)。

    返回:
    - dict: 包含计算结果的字典。
    """
    intra_item_scores_with_self = []
    intra_item_scores_without_self = []
    inter_item_scores = []

    # 从 token 1 开始计算，所以有效 item token 数量为 seq_len - 1
    num_items = (seq_len - 1) // 4

    # 如果 item 不足两个，则无法计算 item 间注意力
    if num_items < 2:
        # 仍然可以计算单个 item 的内部注意力
        if num_items == 1:
            start_idx = 1
            end_idx = min(1 + 4, seq_len)
            item_indices = list(range(start_idx, end_idx))
            if len(item_indices) > 1:
                for q_idx in item_indices:
                    for k_idx in item_indices:
                        score = attention_matrix[q_idx, k_idx]
                        intra_item_scores_with_self.append(score)
                        if q_idx != k_idx:
                            intra_item_scores_without_self.append(score)
        
        avg_intra_with_self = np.mean(intra_item_scores_with_self) if intra_item_scores_with_self else 0
        avg_intra_without_self = np.mean(intra_item_scores_without_self) if intra_item_scores_without_self else 0
        
        return {
            "avg_intra_item_score_with_self": avg_intra_with_self,
            "avg_intra_item_score_without_self": avg_intra_without_self,
            "avg_inter_item_score": 0,
            "comparison_with_self": "N/A (不足2个item)",
            "comparison_without_self": "N/A (不足2个item)",
            "intra_item_scores_with_self_count": len(intra_item_scores_with_self),
            "intra_item_scores_without_self_count": len(intra_item_scores_without_self),
            "inter_item_scores_count": 0
        }

    # 预先计算所有 item 的索引范围
    item_indices_list = []
    for i in range(num_items):
        start_idx = 1 + 4 * i
        end_idx = min(start_idx + 4, seq_len)
        item_indices_list.append(list(range(start_idx, end_idx)))

    # 遍历每个 item 作为 "源" item (从中选取 query)
    for k in range(num_items):
        query_item_indices = item_indices_list[k]
        
        # 遍历所有 item 作为 "目标" item (从中选取 key)
        for j in range(num_items):
            key_item_indices = item_indices_list[j]

            # --- 根据源和目标是否相同，分配到不同列表 ---
            if k == j:  # Intra-item (内部) 注意力
                if len(query_item_indices) < 2: continue
                for q_idx in query_item_indices:
                    for k_idx in key_item_indices:
                        score = attention_matrix[q_idx, k_idx]
                        # 版本1: 包含对自身的注意力
                        intra_item_scores_with_self.append(score)
                        # 版本2: 不包含对自身的注意力
                        if q_idx != k_idx:
                            intra_item_scores_without_self.append(score)
            else:  # Inter-item (之间) 注意力
                for q_idx in query_item_indices:
                    for k_idx in key_item_indices:
                        inter_item_scores.append(attention_matrix[q_idx, k_idx])

    # --- 计算平均值 ---
    avg_intra_with_self = np.mean(intra_item_scores_with_self) if intra_item_scores_with_self else 0
    avg_intra_without_self = np.mean(intra_item_scores_without_self) if intra_item_scores_without_self else 0
    avg_inter = np.mean(inter_item_scores) if inter_item_scores else 0

    # --- 进行比较 ---
    def get_comparison(score1, score2, name1, name2):
        if score1 > score2: return f"{name1} > {name2}"
        if score2 > score1: return f"{name2} > {name1}"
        return f"{name1} == {name2}"

    comp_with_self = get_comparison(avg_intra_with_self, avg_inter, "平均内部注意力(含自身)", "平均跨item注意力")
    comp_without_self = get_comparison(avg_intra_without_self, avg_inter, "平均内部注意力(不含自身)", "平均跨item注意力")

    return {
        "avg_intra_item_score_with_self": avg_intra_with_self,
        "avg_intra_item_score_without_self": avg_intra_without_self,
        "avg_inter_item_score": avg_inter,
        "comparison_with_self": comp_with_self,
        "comparison_without_self": comp_without_self,
        "intra_item_scores_with_self_count": len(intra_item_scores_with_self),
        "intra_item_scores_without_self_count": len(intra_item_scores_without_self),
        "inter_item_scores_count": len(inter_item_scores)
    }
def calculate_token_level_attention_v3(attention_matrix, seq_len):
    """
    以"Token级平均"的视角，计算三种注意力指标。

    1.  Avg Intra-Item: 对每个token，计算它对自身item内其他token的平均注意力，
        然后将这些平均值在所有token上再次平均。
    2.  Avg Inter-Item: 对每个token，计算它对所有外部items内所有token的平均注意力，
        然后将这些平均值在所有token上再次平均。
    3.  Avg Attention To Each Item: 对于每个目标item J，计算所有token对J内所有token的
        平均注意力。这会返回一个列表，每个元素对应一个目标item。

    参数:
    - attention_matrix (np.array): (seq_len, seq_len) 的注意力权重矩阵。
    - seq_len (int): 序列的有效长度 (不包含 padding)。

    返回:
    - dict: 包含三种分析结果的字典。
    """
    if seq_len <= 1:
        return {}

    num_items = (seq_len - 1) // 4
    if num_items == 0:
        return {}

    # --- 步骤1: 准备辅助数据结构 ---
    item_indices_list = []
    token_to_item_map = {}  # 创建一个从token索引到其item索引的映射
    for i in range(num_items):
        start_idx = 1 + 4 * i
        end_idx = min(start_idx + 4, seq_len)
        indices = list(range(start_idx, end_idx))
        item_indices_list.append(indices)
        for token_idx in indices:
            token_to_item_map[token_idx] = i

    all_item_tokens_indices = list(range(1, seq_len))
    if not all_item_tokens_indices:
        return {}
        
    # --- 步骤2: 计算 Token级 Intra-和 Inter- 注意力 ---
    per_token_avg_intra_scores = []
    per_token_avg_inter_scores = []

    # 从每个query_token的视角出发
    for q_idx in all_item_tokens_indices:
        q_item_idx = token_to_item_map[q_idx]
        
        # 找出当前token的内部和外部key token
        intra_k_indices = [k for k in item_indices_list[q_item_idx] if k != q_idx]
        inter_k_indices = [k for k in all_item_tokens_indices if token_to_item_map[k] != q_item_idx]

        # 计算对内的平均注意力
        if intra_k_indices:
            avg_intra = np.mean([attention_matrix[q_idx, k_idx] for k_idx in intra_k_indices])
            per_token_avg_intra_scores.append(avg_intra)
        
        # 计算对外的平均注意力
        if inter_k_indices:
            avg_inter = np.mean([attention_matrix[q_idx, k_idx] for k_idx in inter_k_indices])
            per_token_avg_inter_scores.append(avg_inter)

    # 在所有token上求最终平均
    final_avg_intra = np.mean(per_token_avg_intra_scores) if per_token_avg_intra_scores else 0
    final_avg_inter = np.mean(per_token_avg_inter_scores) if per_token_avg_inter_scores else 0
    
    # --- 步骤3: 计算对每个特定Item的平均注意力 ---
    attention_to_each_item = []
    for j in range(num_items): # 遍历每个item作为"目标"
        target_item_indices = item_indices_list[j]
        
        # 计算所有源token到这个目标item的平均注意力
        all_scores_to_item_j = []
        for q_idx in all_item_tokens_indices:
            # 不区分q_idx是否在目标item内，计算它对目标item的注意力
            scores = [attention_matrix[q_idx, k_idx] for k_idx in target_item_indices]
            if scores:
                 all_scores_to_item_j.extend(scores)

        avg_attention_to_j = np.mean(all_scores_to_item_j) if all_scores_to_item_j else 0
        attention_to_each_item.append({
            "target_item_index": j,
            "avg_attention_score": avg_attention_to_j
        })

    return {
        "token_level_avg_intra_item_attention (without_self)": final_avg_intra,
        "token_level_avg_inter_item_attention": final_avg_inter,
        "avg_attention_to_each_item": attention_to_each_item,
        "comparison": f"Intra ({final_avg_intra:.4f}) {' >' if final_avg_intra > final_avg_inter else ' <'} Inter ({final_avg_inter:.4f})"
    }
import json
def extract_and_save_vectors(model, output_file_path="vocab_vectors.txt"):
    """
    (更健壮的版本) 从模型中提取词汇表的所有向量，并以文本格式保存。
    此版本不依赖 tokenizer，直接通过 embedding 层的 shape 进行遍历，
    适用于 token 就是其整数 ID 的情况。

    Args:
        model: 你训练好的模型对象。
        output_file_path (str): 保存向量文件的路径。
    """
    print("🚀 开始提取词汇表向量 (健壮模式)...")
    
    # 1. 获取模型的词嵌入层
    try:
        word_embeddings = model.get_input_embeddings()
        if word_embeddings is None:
            print("❌ 错误：无法从模型中获取词嵌入层 (model.get_input_embeddings() is None)。")
            return
    except Exception as e:
        print(f"❌ 获取词嵌入层时出错: {e}")
        return

    # 2. 获取嵌入矩阵，并转移到CPU
    embedding_matrix = word_embeddings.weight.detach().cpu().numpy()
    
    # 3. 直接从矩阵形状获取词汇量和维度
    num_embeddings, embedding_dim = embedding_matrix.shape
    print(f"检测到词汇量: {num_embeddings}, 向量维度: {embedding_dim}")

    # 4. 将向量写入文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write(f"{num_embeddings} {embedding_dim}\n")
            
            # 直接按索引 (token_id) 遍历
            for token_id in range(num_embeddings):
                vector = embedding_matrix[token_id]
                vector_str = ' '.join(map(str, vector))
                # 将 token_id 本身作为 "token" 写入
                f.write(f"{token_id} {vector_str}\n")
        
        print(f"✅ 成功将 {num_embeddings} 个向量写入到: '{os.path.abspath(output_file_path)}'")

    except IOError as e:
        print(f"❌ 写入文件时发生IO错误: {e}")
    except Exception as e:
        print(f"❌ 写入过程中发生未知错误: {e}")
from itertools import combinations
def _get_historical_items(input_ids_tensor, tokenizer):
    """
    辅助函数，用于将输入的 token ID 序列解析成历史 item 列表。
    新规则:
    1. 移除所有 padding tokens.
    2. 序列中的第一个有效 token 是 user token，将被忽略.
    3. 之后每 4 个 token 构成一个 item.
    """
    pad_token_id = 0

    # 1. 转换为列表并移除 padding
    tokens = [t for t in input_ids_tensor.tolist() if t != pad_token_id]

    # 2. 检查是否有足够的 token (至少1个user token + 4个item token)，并移除 user token
    if len(tokens) < 5: # 至少需要 user token + 1个item
        return []

    # 忽略第一个 token (user token)
    item_tokens = tokens[1:]

    # 3. 将剩余的 token 每 4 个一组进行切分
    history = []
    num_item_tokens = len(item_tokens)
    for i in range(0, num_item_tokens, 4):
        # 确保我们有一个完整的 4-token item
        if i + 4 <= num_item_tokens:
            history.append(item_tokens[i:i + 4])
            
    return history

import csv
def calculate_and_log_attention_stats(
    attention_matrix, 
    input_ids, 
    tokenizer, 
    output_csv_path, 
    item_size=4,
    pad_token_id=0
):
    """
    计算并记录注意力统计（最终修正版，已适配左padding，带调试信息）。
    """
    print(f"   [调试] 接收到 (左padding) input_ids (前30个): {input_ids[:30]}")

    # --- 核心改动：适配左padding ---
    # 从头开始遍历，找到第一个“非”padding token的索引，这才是有效序列的开始
    valid_start_index = -1
    for i, token_id in enumerate(input_ids):
        if token_id != pad_token_id:
            valid_start_index = i
            break
    
    # --- 调试：打印计算出的序列信息 ---
    print(f"   [调试] 第一个有效token的索引 (valid_start_index): {valid_start_index}")

    # 如果没有找到任何有效token（整个序列都是padding），或只有一个token
    if valid_start_index == -1 or (len(input_ids) - valid_start_index) <= 1:
        real_seq_len = 0 if valid_start_index == -1 else (len(input_ids) - valid_start_index)
        print(f"   [调试] 真实序列长度 <= 1 (长度为 {real_seq_len})，无法进行item分析，函数提前返回(0,0,0)。")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['error'])
            writer.writerow([f'No valid tokens to analyze. Real sequence length was {real_seq_len}.'])
        return 0, 0, 0

    # 根据有效序列的起点，对数据进行切片
    valid_input_ids = input_ids[valid_start_index:]
    real_seq_len = len(valid_input_ids)
    
    if hasattr(attention_matrix, 'cpu'):
        attention_matrix_np = attention_matrix.cpu().numpy()
    else:
        attention_matrix_np = attention_matrix
        
    # 注意力矩阵也要进行同样的切片
    valid_attention_matrix = attention_matrix_np[valid_start_index:, valid_start_index:]
    
    print(f"   [调试] 切片后的真实序列长度: {real_seq_len}")
    print(f"   [调试] 切片后的 attention_matrix 形状: {valid_attention_matrix.shape}")

    token_stats = []
    
    # 偏移量仍然是1，因为我们要跳过有效序列中的第一个token（user token）
    offset = 1

    # 在“有效序列”的范围内进行遍历
    for query_idx in range(offset, real_seq_len):
        query_token_id = valid_input_ids[query_idx]
        query_token = query_token_id

        item_index = (query_idx - offset) // item_size
        item_start_idx = offset + item_index * item_size
        item_end_idx = min(item_start_idx + item_size, real_seq_len)

        own_item_scores_with_self = []
        own_item_scores_no_self = []
        other_item_scores = []

        for key_idx in range(offset, real_seq_len):
            score = valid_attention_matrix[query_idx, key_idx]

            if item_start_idx <= key_idx < item_end_idx:
                own_item_scores_with_self.append(score)
                if query_idx != key_idx:
                    own_item_scores_no_self.append(score)
            else:
                other_item_scores.append(score)
        
        avg_own_with_self = np.mean(own_item_scores_with_self) if own_item_scores_with_self else 0
        avg_own_no_self = np.mean(own_item_scores_no_self) if own_item_scores_no_self else 0
        avg_other = np.mean(other_item_scores) if other_item_scores else 0
        
        token_stats.append({
            # 注意：这里的索引是相对于有效序列的，而不是原始序列
            'token_index_in_valid_seq': query_idx,
            'token': query_token,
            'item_index': item_index,
            'avg_attention_to_own_item (with self)': f"{avg_own_with_self:.6f}",
            'avg_attention_to_own_item (no self)': f"{avg_own_no_self:.6f}",
            'avg_attention_to_other_items': f"{avg_other:.6f}"
        })

    if not token_stats:
        print("   [调试] 循环结束但未收集到任何token统计数据，返回(0,0,0)。")
        return 0, 0, 0
            
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=token_stats[0].keys())
        writer.writeheader()
        writer.writerows(token_stats)
        
    total_avg_own_with_self = np.mean([float(s['avg_attention_to_own_item (with self)']) for s in token_stats])
    total_avg_own_no_self = np.mean([float(s['avg_attention_to_own_item (no self)']) for s in token_stats])
    total_avg_other = np.mean([float(s['avg_attention_to_other_items']) for s in token_stats])

    return total_avg_own_with_self, total_avg_own_no_self, total_avg_other

vis_config = {
    "shared_prefix_1": 5,
    "shared_prefix_2": 5,
    "shared_prefix_3": 5,
    "shared_token_no_prefix": 5,
    "other_samples": 5
}