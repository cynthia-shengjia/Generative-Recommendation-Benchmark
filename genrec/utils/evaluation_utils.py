import os
import torch
import math
import json
from tqdm import tqdm
from genrec.utils.metrics import tokens_to_item_id
from genrec.generation.trie import Trie,prefix_allowed_tokens_fn


def evaluate_model_with_constrained_beam_search(
    model,
    eval_dataloader,
    accelerator,
    tokenizer,
    k_list=[1, 5, 10],
    num_beams=10,
    max_gen_length=5,
    logger=None,
    mode="Evaluation",
    visualization_config=None,
    visualization_dir="attention_heatmaps",
    output_json_path="predictions.json",
    vector_output_path="vocab_vectors.txt",
):
    """
    评估模型或根据特定标准可视化注意力。
    - visualization_config (dict): 激活可视化模式，并指定各类别的样本数量。
        例如: {'shared_prefix_1': 5, 'shared_prefix_2': 5, 'shared_token_no_prefix': 3}
    """
    model.eval()

    # =================================================================================
    # 可视化逻辑
    # =================================================================================
    if visualization_config and accelerator.is_main_process:
        print("进入条件可视化模式...")
        os.makedirs(visualization_dir, exist_ok=True)
        print(f"热力图将分类保存到: '{visualization_dir}/'")

        required_counts = visualization_config.copy()
        samples_to_visualize = {} 

        print("第一步: 遍历数据集以筛选符合条件的样本 (基于历史记录内部比较)...")
        # 筛选 pass
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="筛选样本")):
            if all(v == 0 for v in required_counts.values()):
                print("已为所有类别找到足够数量的样本。")
                break 

            input_ids_cpu = batch['input_ids']

            for i in range(input_ids_cpu.size(0)):
                if all(v == 0 for v in required_counts.values()):
                    break
                
                # +++++ 全新的筛选逻辑 +++++
                historical_items = _get_historical_items(input_ids_cpu[i], tokenizer)
                if len(historical_items) < 2:
                    continue

                pairs = list(combinations(historical_items, 2))

                # --- Pass 1: 检查整个历史中最高级别的前缀匹配 ---
                best_prefix_level = 0
                for item1, item2 in pairs:
                    if len(item1) >= 3 and len(item2) >= 3 and item1[:3] == item2[:3]:
                        best_prefix_level = max(best_prefix_level, 3)
                    elif len(item1) >= 2 and len(item2) >= 2 and item1[:2] == item2[:2]:
                        best_prefix_level = max(best_prefix_level, 2)
                    elif len(item1) >= 1 and len(item2) >= 1 and item1[:1] == item2[:1]:
                        best_prefix_level = max(best_prefix_level, 1)

                found_category = None
                if best_prefix_level == 3:
                    found_category = 'shared_prefix_3'
                elif best_prefix_level == 2:
                    found_category = 'shared_prefix_2'
                elif best_prefix_level == 1:
                    found_category = 'shared_prefix_1'
                else:
                    # --- Pass 2: 仅当历史中没有任何前缀匹配时，才检查共享token ---
                    has_shared_token = False
                    for item1, item2 in pairs:
                        if not set(item1[:3]).isdisjoint(set(item2[:3])):
                            has_shared_token = True
                            break
                    if has_shared_token:
                        found_category = 'shared_token_no_prefix'
                if found_category is None:
                    found_category = 'other_samples'
                # 如果找到了符合条件的类别，并且该类别还需要样本，则记录
                if found_category and required_counts.get(found_category, 0) > 0:
                    sample_key = (batch_idx, i)
                    if sample_key not in samples_to_visualize:
                        samples_to_visualize[sample_key] = found_category
                        required_counts[found_category] -= 1
                        print(f"  [找到样本] Batch {batch_idx}, Sample {i} -> 分类到 '{found_category}'. 该类别还需 {required_counts[found_category]} 个.")

        # --- 将筛选结果重组，方便后续处理 ---
        batches_to_process = {}
        for (batch_idx, sample_idx), category in samples_to_visualize.items():
            if batch_idx not in batches_to_process:
                batches_to_process[batch_idx] = []
            batches_to_process[batch_idx].append((sample_idx, category))

        if not batches_to_process:
            print("未能找到任何符合可视化条件的样本。")
            return

        print(f"\n筛选完成！共找到 {len(samples_to_visualize)} 个样本用于可视化。")
        print("第二步: 开始对筛选出的样本进行可视化...")

        # 可视化 pass (这部分逻辑不变)
        device = accelerator.device
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="生成热力图")):
                if batch_idx not in batches_to_process:
                    continue

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, max_length=2,
                    num_beams=1, return_dict_in_generate=True, output_attentions=True
                )
                if not (hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions is not None):
                    print("错误: 模型输出中未找到 'encoder_attentions'。")
                    return
                
                last_layer_attentions = outputs.encoder_attentions[-1]
                
                for sample_idx, category in batches_to_process[batch_idx]:
                    print(f"\n--- 正在可视化 Batch {batch_idx}, Sample {sample_idx} (类别: {category}) ---")
                    
                    sample_attentions = last_layer_attentions[sample_idx]
                    sample_input_ids = input_ids[sample_idx].cpu().tolist()
                    
                    sample_output_dir = os.path.join(visualization_dir, category, f"batch_{batch_idx}_sample_{sample_idx}")
                    os.makedirs(sample_output_dir, exist_ok=True)
                    
                    avg_attention_weights = sample_attentions.mean(dim=0)
                    avg_file_path = os.path.join(sample_output_dir, "heatmap_average.png")
                    plot_encoder_attention_heatmap(avg_attention_weights, sample_input_ids, file_path=avg_file_path)
                    print(f"   [统计] 正在为样本计算注意力统计数据...")
                    stats_csv_path = os.path.join(sample_output_dir, "attention_statistics.csv")
                    
                    # 使用平均注意力矩阵进行计算
                    total_avg_own, total_avg_own_no_self, total_avg_other = calculate_and_log_attention_stats(
                        attention_matrix=avg_attention_weights,
                        input_ids=sample_input_ids,
                        tokenizer=tokenizer,
                        output_csv_path=stats_csv_path,
                        item_size=4  # 根据您的定义，每个item是4个token
                    )
                    
                    print(f"   [统计] 详细数据已保存到: {stats_csv_path}")
                    print(f"   [统计] 样本总体平均注意力:")
                    print(f"     - 对本Item (含自身): {total_avg_own:.6f}")
                    print(f"     - 对本Item (不含自身): {total_avg_own_no_self:.6f}")
                    print(f"     - 对其他Item: {total_avg_other:.6f}")
                    num_heads = sample_attentions.shape[0]
                    for head_idx in range(num_heads):
                        head_attention_weights = sample_attentions[head_idx]
                        file_path = os.path.join(sample_output_dir, f"head_{head_idx}_heatmap.png")
                        plot_encoder_attention_heatmap(
                            head_attention_weights,
                            sample_input_ids,
                            file_path=file_path,
                            head_index=head_idx
                        )
        
        print(f"\n已成功为 {len(samples_to_visualize)} 个筛选出的样本生成可视化图像。")
        return

    if visualization_config:
         accelerator.wait_for_everyone() # 确保所有进程同步
         if accelerator.is_main_process:
            print("可视化任务已在主进程完成，跳过评估部分。")
         return
         
    tokens_to_item_map = tokenizer.tokens2item
    item_to_tokens_map = tokenizer.item2tokens
    candidate_trie = Trie(tokenizer.item2tokens)
    prefix_allowed_fn = prefix_allowed_tokens_fn(candidate_trie)
    all_predictions = []
    all_labels = []
    device = accelerator.device
    progress_bar = tqdm(eval_dataloader, desc=f"{mode}", disable=not accelerator.is_main_process)
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        true_item_ids = batch['label_id']
        # encoder_item_ids = batch['encoder_item_ids'].to(device) 
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_length,
            # encoder_item_ids=encoder_item_ids,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
            pad_token_id=0,
            eos_token_id=1,
            output_scores=True,
            return_dict_in_generate=True,
            prefix_allowed_tokens_fn=prefix_allowed_fn
        )
        generated_ids = outputs.sequences
        sequences_scores = outputs.sequences_scores
        batch_size = input_ids.size(0)
        generated_ids_reshaped = generated_ids.view(batch_size, num_beams, -1)[:, :, 1:]
        probabilities_reshaped = torch.exp(sequences_scores).view(batch_size, num_beams)
        batch_predictions = []
        batch_labels = []
        for i in range(batch_size):
            true_item_id = true_item_ids[i]
            true_item_tokens = item_to_tokens_map.get(true_item_id, []) 
            batch_labels.append({
                'id': true_item_id,
                'tokens': true_item_tokens
            })
            user_sequences = generated_ids_reshaped[i]
            user_probs = probabilities_reshaped[i]
            sorted_indices = torch.argsort(user_probs, descending=True)
            sorted_sequences = user_sequences[sorted_indices]
            predictions_with_tokens = []
            for seq in sorted_sequences:
                pred_tokens = seq.tolist()
                item_id = tokens_to_item_id(pred_tokens, tokens_to_item_map)
                predictions_with_tokens.append({
                    'id': item_id,
                    'tokens': pred_tokens
                })
            unique_predictions = []
            seen_ids = set()
            for pred in predictions_with_tokens:
                if pred['id'] not in seen_ids:
                    unique_predictions.append(pred)
                    seen_ids.add(pred['id'])
            batch_predictions.append(unique_predictions)
        gathered_predictions = accelerator.gather_for_metrics(batch_predictions)
        gathered_labels = accelerator.gather_for_metrics(batch_labels)
        all_predictions.extend(gathered_predictions)
        all_labels.extend(gathered_labels)

    if accelerator.is_main_process:
        metrics = {f'hit@{k}': 0 for k in k_list}
        metrics.update({f'ndcg@{k}': 0 for k in k_list})
        total_samples = len(all_labels)
        results_to_save = []
        for label_info, prediction_list in zip(all_labels, all_predictions):
            
            true_item_id = label_info['id']
            predicted_item_ids = [p['id'] for p in prediction_list]

            for k in k_list:
                top_k_ids = predicted_item_ids[:k]
                hit = 1 if true_item_id in top_k_ids else 0
                metrics[f'hit@{k}'] += hit
                if hit:
                    rank = top_k_ids.index(true_item_id) + 1
                    ndcg = 1 / math.log2(rank + 1)
                    metrics[f'ndcg@{k}'] += ndcg
            
            sample_result = {
                'label_id': label_info['id'],
                'label_tokens': label_info['tokens'],
                'predictions': [
                    {
                        'predicted_id': p['id'],
                        'predicted_tokens': p['tokens']
                    } for p in prediction_list
                ]
            }
            results_to_save.append(sample_result)
        final_metrics = {}
        for key, value in metrics.items():
            final_metrics[key] = value / total_samples

        if logger:
            logger.info(f"\n{mode} 结果 (共 {total_samples} 个样本):")
            for k in k_list:
                logger.info(f"Hit@{k}: {final_metrics[f'hit@{k}']:.4f}, NDCG@{k}: {final_metrics[f'ndcg@{k}']:.4f}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        
        if logger:
            logger.info(f"Top-10 预测结果已保存到: {output_json_path}")
        return final_metrics
