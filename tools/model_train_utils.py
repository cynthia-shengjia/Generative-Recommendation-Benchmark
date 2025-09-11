import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from genrec.models.CustomT5.T5Model import CustomT5ForConditionalGeneration
from genrec.models.CustomT5.T5Config import CustomT5Config
from genrec.tokenizers.TigerTokenizer import TigerTokenizer
from tools.utils import calc_ndcg, tokens_to_item_id
from typing import List, Dict, Any
import math
import logging
import re
from transformers import T5ForConditionalGeneration,T5Config,Trainer
import numpy as np
import math
from transformers import EvalPrediction

def compute_metrics(p: EvalPrediction, tokens_to_item_map: dict, k_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    计算评估指标的函数，用于 Hugging Face Trainer。
    这个函数假设 p.predictions 是模型生成的结果，p.label_ids 是真实标签。

    参数:
        p (EvalPrediction): 包含 predictions 和 label_ids 的对象。
                            - predictions: (num_samples, num_beams, seq_len) 的 numpy 数组，是生成的 token ID。
                            - label_ids: (num_samples,) 的 numpy 数组，是真实的 item ID。
        k_list (List[int]): 需要计算指标的 K 值列表。

    返回:
        Dict[str, float]: 包含所有指标的字典。
    """
    # p.predictions 的形状是 (batch_size, num_beams, max_gen_length)
    # p.label_ids 的形状是 (batch_size, 1) 或 (batch_size,)

    batch_size = p.predictions.shape[0]
    num_beams = p.predictions.shape[1]

    generated_ids = p.predictions
    # generated_ids 是我们通过自定义 Trainer 传过来的生成结果
    generated_ids_tensor = torch.from_numpy(p.predictions)
    generated_ids_reshaped = generated_ids_tensor.view(batch_size, num_beams, -1)[:, :, 1:]
    
    all_predictions = []
    all_labels = []
    for label_group in p.label_ids:
        token_ids_for_lookup = label_group[:-1]
        
        true_item_id = tokens_to_item_id(
            token_ids_for_lookup.tolist(),
            tokens_to_item_map
        )
        
        all_labels.append(true_item_id)
    for i in range(batch_size):
        user_sequences = generated_ids_reshaped[i] # (num_beams, max_gen_length)
        
        # beam search 的结果已经按分数排好序，
        item_ids = []
        for seq in user_sequences:
            item_id = tokens_to_item_id(
                seq.tolist(),
                tokens_to_item_map
            )
            item_ids.append(item_id)
        
        # 去重并保持顺序
        seen = set()
        unique_item_ids = [x for x in item_ids if x is not None and not (x in seen or seen.add(x))]
        all_predictions.append(unique_item_ids)

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

    # 计算平均指标
    final_metrics = {}
    for key, value in metrics.items():
        final_metrics[key] = value / total_samples
    return final_metrics
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

def create_custom_t5_model(
    vocab_size: int,
    d_kv: int = 64,         # 每个注意力头的维度
    d_ff: int = 1024,       # 前馈网络维度
    num_layers: int = 4,    # 编码器层数
    num_decoder_layers: int = 4,  # 解码器层数
    num_heads: int = 6,     # 注意力头数
    dropout_rate: float = 0.1,  # dropout率
    tie_word_embeddings: bool = True
) -> CustomT5ForConditionalGeneration:
    """
    创建自定义T5模型，根据提供的配置参数
    d_model 会自动计算为 d_kv * num_heads
    """
    config = CustomT5Config(
        vocab_size=vocab_size,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        tie_word_embeddings=tie_word_embeddings
    )
    
    # 计算 latent_dim = d_model = d_kv * num_heads
    latent_dim = d_kv * num_heads
    
    model = CustomT5ForConditionalGeneration(
        config=config,
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        tie_word_embeddings=tie_word_embeddings
    )
    
    return model

def train_model(
    model: CustomT5ForConditionalGeneration,
    train_dataloader: torch.utils.data.DataLoader,
    learning_rate: float,
    num_epochs: int,
    checkpoint_dir: str,
    dataset_name: str,
    accelerator: Accelerator,
    log_interval: int = 100,
    logger: logging.Logger = None,
    num_steps: int = None,
) -> CustomT5ForConditionalGeneration:
    """
    使用Accelerate进行模型训练，并按Epoch保存和加载检查点。
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if num_steps is None:
        total_steps = len(train_dataloader) * num_epochs
    else:
        total_steps = num_steps
        # 如果指定了总步数，num_epochs 仍然用于计算 scheduler，但循环将提前退出
        num_epochs = (total_steps // len(train_dataloader)) + 1 if len(train_dataloader) > 0 else 1

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    full_checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
    os.makedirs(full_checkpoint_dir, exist_ok=True)
    
    start_epoch = 0 
    checkpoint_folders = [d for d in os.listdir(full_checkpoint_dir) if d.startswith('epoch_')]
    if checkpoint_folders:
        latest_epoch = max([int(re.search(r'epoch_(\d+)', d).group(1)) for d in checkpoint_folders])
        latest_checkpoint_path = os.path.join(full_checkpoint_dir, f"epoch_{latest_epoch}")
        if accelerator.is_main_process and latest_checkpoint_path:
            logger.info(f"发现最新检查点，从 {latest_checkpoint_path} 恢复训练...")
            accelerator.load_state(latest_checkpoint_path)
            start_epoch = latest_epoch 
            logger.info(f"成功恢复！将从 epoch {start_epoch + 1} 开始训练。")

    
    global_step = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not accelerator.is_main_process)
        
        for batch in progress_bar:
            if num_steps is not None and global_step >= num_steps:
                break

            outputs = model(
                encoder_input_ids               = batch['encoder_input_ids'],
                encoder_attention_mask          = batch['encoder_attention_mask'],
                decoder_input_ids               = batch['decoder_input_ids'],
                decoder_attention_mask          = batch['decoder_attention_mask'],
                labels                          = batch['labels']
            )
            
            loss = outputs['loss']
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            if accelerator.is_main_process:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'step': global_step})

        accelerator.wait_for_everyone()
        if (epoch + 1) % 25 == 0:
            save_path = os.path.join(full_checkpoint_dir, f"epoch_{epoch + 1}")
            accelerator.save_state(save_path)
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1} 完成。检查点已保存至: {save_path}")

        if num_steps is not None and global_step >= num_steps and accelerator.is_main_process:
            logger.info(f"已达到目标步数 {num_steps}，训练提前结束。")
            break

    if accelerator.is_main_process:
        logger.info("\n训练完成。")
    return accelerator.unwrap_model(model)




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
    
