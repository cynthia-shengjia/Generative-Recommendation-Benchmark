import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from genrec.tokenizers.TigerTokenizer import TigerTokenizer
from tools.utils import calc_ndcg, tokens_to_item_id
from typing import Optional, Dict, Any, List, Union, Callable  # 添加了必要的导入
import math
import logging
import re
from transformers import T5ForConditionalGeneration,T5Config,Trainer
from genrec.models.CustomMSL.MSLBeamModeling import CustomT5ForConditionalGeneration
import numpy as np
import math
from transformers import EvalPrediction
from functools import partial
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import TrainerCallback, TrainingArguments, TrainerState
from trainers.model_trainers.GenerativeTrainer import GenerativeTrainer
from tools.nni_utils import get_nni_params, update_config_with_nni, report_nni_metrics
from genrec.cbs_structure.generate_trie import Trie,prefix_allowed_tokens_fn

def create_t5_model(vocab_size: int, model_config: dict):
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
    
    model = CustomT5ForConditionalGeneration(config)
    return model





def evaluate_model_with_constrained_beam_search(
    model,
    eval_dataloader,
    accelerator,
    tokenizer,
    k_list=[1, 5, 10],
    num_beams=10,
    max_gen_length=5,
    logger=None,
    mode="Validation"
):
    """
    使用约束beam search评估模型并计算NDCG@K和Hit@K指标
    
    参数:
        model: 已训练的模型
        eval_dataloader: 评估数据加载器
        accelerator: Accelerator实例
        tokenizer: 分词器，包含tokens2item和item2tokens映射
        k_list: 要计算的K值列表
        num_beams: beam search的beam数量
        max_gen_length: 生成序列的最大长度
        logger: 日志记录器
        mode: 评估模式（Validation或Test）
    
    返回:
        包含各项指标的字典
    """
    model.eval()
    
    # 初始化Trie结构和约束函数
    tokens_to_item_map = tokenizer.tokens2item
    candidate_trie = Trie(tokenizer.item2tokens)
    prefix_allowed_fn = prefix_allowed_tokens_fn(candidate_trie)
    
    all_predictions = []
    all_labels = []
    # 获取设备
    device = accelerator.device
    
    with torch.no_grad():
        progress_bar = tqdm(eval_dataloader, desc=f"{mode}", disable=not accelerator.is_main_process)
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 获取真实物品ID
            true_item_ids = batch['label_id']

            
            # 使用约束beam search生成序列
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_length,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True,
                pad_token_id=0,
                eos_token_id=1,
                output_scores=True,
                return_dict_in_generate=True,
                prefix_allowed_tokens_fn=prefix_allowed_fn  # 使用Trie约束
            )
            
            # 获取生成的序列和分数
            generated_ids = outputs.sequences
            sequences_scores = outputs.sequences_scores
            
            # 重塑结果以便按样本分组
            batch_size = input_ids.size(0)
            generated_ids_reshaped = generated_ids.view(batch_size, num_beams, -1)[:, :, 1:]
            probabilities_reshaped = torch.exp(sequences_scores).view(batch_size, num_beams)
            
            batch_predictions = []
            batch_labels = []
            # 处理每个样本
            for i in range(batch_size):
                true_item_id = true_item_ids[i]
                batch_labels.append(true_item_id)
                user_sequences = generated_ids_reshaped[i]
                user_probs = probabilities_reshaped[i]
                
                # 按概率从高到低排序
                sorted_indices = torch.argsort(user_probs, descending=True)
                sorted_sequences = user_sequences[sorted_indices]
                
                # 将tokens转换为item ID
                item_ids = []
                for seq in sorted_sequences:
                    # 跳过特殊token
                    item_id = tokens_to_item_id(
                        seq.tolist(), 
                        tokens_to_item_map
                    )
                    item_ids.append(item_id)

                
                # 去重并保持顺序
                seen = set()
                item_ids = [x for x in item_ids if not (x in seen or seen.add(x))]
                batch_predictions.append(item_ids)
            gathered_predictions = accelerator.gather_for_metrics(batch_predictions)
            gathered_labels = accelerator.gather_for_metrics(batch_labels)

            all_predictions.extend(gathered_predictions)
            all_labels.extend(gathered_labels)

    if accelerator.is_main_process:
        metrics = {f'hit@{k}': 0 for k in k_list}
        metrics.update({f'ndcg@{k}': 0 for k in k_list})
        total_samples = len(all_labels)

        # 在所有收集到的数据上进行计算
        for true_item_id, predicted_items in zip(all_labels, all_predictions):
            for k in k_list:
                top_k_items = predicted_items[:k]
                
                hit = 1 if true_item_id in top_k_items else 0
                metrics[f'hit@{k}'] += hit
                
                if hit:
                    rank = top_k_items.index(true_item_id) + 1
                    ndcg = 1 / math.log2(rank + 1)
                    metrics[f'ndcg@{k}'] += ndcg
        
        # 计算平均指标
        final_metrics = {}
        for key, value in metrics.items():
            final_metrics[key] = value / total_samples

        if logger:
            logger.info(f"\n{mode} 结果 (共 {total_samples} 个样本):")
            for k in k_list:
                logger.info(f"Hit@{k}: {final_metrics[f'hit@{k}']:.4f}, NDCG@{k}: {final_metrics[f'ndcg@{k}']:.4f}")
        
        return final_metrics 
    

