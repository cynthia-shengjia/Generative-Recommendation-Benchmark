import torch
import math
import math
from transformers import EvalPrediction
from typing import Dict, List 
from genrec.utils.common_utils import tokens_to_item_id

def calc_ndcg(rank, k):
    if rank <= k:
        return 1 / math.log2(rank + 1)
    return 0

def compute_metrics(p: EvalPrediction, tokens_to_item_map: dict, k_list: List[int] = [1, 5, 10]) -> Dict[str, float]:

    batch_size = p.predictions.shape[0]
    num_beams = p.predictions.shape[1]

    generated_ids_tensor = torch.from_numpy(p.predictions)

    generated_ids_reshaped = generated_ids_tensor.view(batch_size, num_beams, -1)[:, :, 1:]
    
    all_predictions = []
    for i in range(batch_size):
        user_sequences = generated_ids_reshaped[i]
        
        item_ids = []
        for seq in user_sequences:
            tokens_tuple = tuple(seq.tolist())
            item_id = tokens_to_item_map.get(tokens_tuple, None)
            item_ids.append(item_id)
        

        seen = set()
        unique_item_ids = [x for x in item_ids if x is not None and not (x in seen or seen.add(x))]
        all_predictions.append(unique_item_ids)


    all_labels = p.label_ids.reshape(-1).tolist() 


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

    final_metrics = {}
    for key, value in metrics.items():
        final_metrics[key] = value / total_samples
        
    return final_metrics
