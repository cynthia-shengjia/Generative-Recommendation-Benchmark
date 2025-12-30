import torch
import math
import math
from transformers import EvalPrediction
from typing import Dict, List 
from genrec.utils.common_utils import tokens_to_item_id

def calc_ndcg(rank, k):
    """
    计算NDCG值
    
    参数:
        rank: 真实物品在推荐列表中的位置(从1开始)
        k: 推荐列表长度
    
    返回:
        NDCG值
    """
    if rank <= k:
        return 1 / math.log2(rank + 1)
    return 0

def compute_metrics(p: EvalPrediction, tokens_to_item_map: dict, k_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    计算评估指标。
    1. label_ids 直接取自原始 Item ID。
    2. predictions 通过 tokens_to_item_map 进行 1:1 映射。
    """
    batch_size = p.predictions.shape[0]
    num_beams = p.predictions.shape[1]

    generated_ids_tensor = torch.from_numpy(p.predictions)
    # 截断起始符 [:, :, 1:]，确保与 map 的 key 长度对齐
    generated_ids_reshaped = generated_ids_tensor.view(batch_size, num_beams, -1)[:, :, 1:]
    
    all_predictions = []
    for i in range(batch_size):
        user_sequences = generated_ids_reshaped[i]
        
        item_ids = []
        for seq in user_sequences:
            tokens_tuple = tuple(seq.tolist())
            item_id = tokens_to_item_map.get(tokens_tuple, None)
            item_ids.append(item_id)
        
        # 去重并保持顺序，同时过滤掉 None
        seen = set()
        unique_item_ids = [x for x in item_ids if x is not None and not (x in seen or seen.add(x))]
        all_predictions.append(unique_item_ids)


    all_labels = p.label_ids.reshape(-1).tolist() 

    # 3. 计算指标
    metrics = {f'hit@{k}': 0.0 for k in k_list}
    metrics.update({f'ndcg@{k}': 0.0 for k in k_list})
    total_samples = len(all_labels)

    for true_item_id, predicted_items in zip(all_labels, all_predictions):
        for k in k_list:
            top_k_items = predicted_items[:k]
            

            hit = 1 if true_item_id in top_k_items else 0
            metrics[f'hit@{k}'] += hit
            
            if hit:
                rank = top_k_items.index(true_item_id) + 1
                metrics[f'ndcg@{k}'] += 1 / math.log2(rank + 1)

    # 4. 计算平均指标
    final_metrics = {}
    for key, value in metrics.items():
        final_metrics[key] = value / total_samples
        
    return final_metrics

# def compute_metrics(p: EvalPrediction, tokens_to_item_map: dict, k_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
#     """
#     计算评估指标的函数，用于 Hugging Face Trainer。
#     这个函数假设 p.predictions 是模型生成的结果，p.label_ids 是真实标签。

#     参数:
#         p (EvalPrediction): 包含 predictions 和 label_ids 的对象。
#                             - predictions: (num_samples, num_beams, seq_len) 的 numpy 数组，是生成的 token ID。
#                             - label_ids: (num_samples,) 的 numpy 数组，是真实的 item ID。
#         k_list (List[int]): 需要计算指标的 K 值列表。

#     返回:
#         Dict[str, float]: 包含所有指标的字典。
#     """
#     # p.predictions 的形状是 (batch_size, num_beams, max_gen_length)
#     # p.label_ids 的形状是 (batch_size, 1) 或 (batch_size,)

#     batch_size = p.predictions.shape[0]
#     num_beams = p.predictions.shape[1]

#     generated_ids = p.predictions
#     # generated_ids 是我们通过自定义 Trainer 传过来的生成结果
#     generated_ids_tensor = torch.from_numpy(p.predictions)
#     generated_ids_reshaped = generated_ids_tensor.view(batch_size, num_beams, -1)[:, :, 1:]
    
#     all_predictions = []
#     all_labels = []
#     for label_group in p.label_ids:
#         token_ids_for_lookup = label_group[:-1]
        
#         true_item_id = tokens_to_item_id(
#             token_ids_for_lookup.tolist(),
#             tokens_to_item_map
#         )
        
#         all_labels.append(true_item_id)
#     for i in range(batch_size):
#         user_sequences = generated_ids_reshaped[i] # (num_beams, max_gen_length)
        
#         # beam search 的结果已经按分数排好序，
#         item_ids = []
#         for seq in user_sequences:
#             item_id = tokens_to_item_id(
#                 seq.tolist(),
#                 tokens_to_item_map
#             )
#             item_ids.append(item_id)
        
#         # 去重并保持顺序
#         seen = set()
#         unique_item_ids = [x for x in item_ids if x is not None and not (x in seen or seen.add(x))]
#         all_predictions.append(unique_item_ids)

#     metrics = {f'hit@{k}': 0.0 for k in k_list}
#     metrics.update({f'ndcg@{k}': 0.0 for k in k_list})
#     total_samples = len(all_labels)

#     for true_item_id, predicted_items in zip(all_labels, all_predictions):
#         for k in k_list:
#             top_k_items = predicted_items[:k]
            
#             hit = 1 if true_item_id in top_k_items else 0
#             metrics[f'hit@{k}'] += hit
            
#             if hit:
#                 rank = top_k_items.index(true_item_id) + 1
#                 metrics[f'ndcg@{k}'] += 1 / math.log2(rank + 1)

#     # 计算平均指标
#     final_metrics = {}
#     for key, value in metrics.items():
#         final_metrics[key] = value / total_samples
#     return final_metrics


