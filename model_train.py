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
from genrec.cbs_structure.generate_trie import Trie, prefix_allowed_tokens_fn
from utils import calc_ndcg, tokens_to_item_id
from typing import List, Dict, Any
import math
import logging
import re
from transformers import T5ForConditionalGeneration,T5Config



def create_t5_model(vocab_size: int, model_config: dict) -> T5ForConditionalGeneration:
    """
    创建标准的T5模型，根据提供的配置参数
    """
    config = T5Config(
        vocab_size=vocab_size,
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


def test_model(
    model: CustomT5ForConditionalGeneration,
    test_dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    tokenizer: TigerTokenizer,
    k_list: List[int] = [1, 5, 10],
    num_beams: int = 10,
    max_gen_length: int = 5,  # 生成序列的最大长度
    logger: logging.Logger = None
) -> Dict[str, float]:
    """
    测试模型并计算NDCG@K和Hit@K指标
    
    参数:
        model: 已训练的模型
        test_dataloader: 测试数据加载器
        accelerator: Accelerator实例
        tokenizer: 分词器，包含tokens2item和item2tokens映射
        k_list: 要计算的K值列表
        num_beams: beam search的beam数量
        max_gen_length: 生成序列的最大长度
        logger: 日志记录器
    
    返回:
        包含各项指标的字典
    """
    model.eval()
    
    # 初始化Trie结构和约束函数
    tokens_to_item_map = tokenizer.tokens2item
    candidate_trie = Trie(tokenizer.item2tokens)
    prefix_allowed_fn = prefix_allowed_tokens_fn(candidate_trie)
    
    # 初始化指标累计器
    metrics = {f'hit@{k}': 0 for k in k_list}
    metrics.update({f'ndcg@{k}': 0 for k in k_list})
    total_samples = 0
    
    # 获取设备
    device = accelerator.device
    
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Testing", disable=not accelerator.is_main_process)
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            encoder_input_ids = batch['encoder_input_ids'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            
            # 获取真实物品ID
            true_item_ids = batch['label_id'].tolist()
            # 使用beam search生成序列
            outputs = model.generate(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                max_length=max_gen_length,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True,
                decoder_start_token_id=0,
                output_scores=True,
                return_dict_in_generate=True,
                prefix_allowed_tokens_fn=prefix_allowed_fn  # 使用Trie约束
            )
            
            # 获取生成的序列和分数
            generated_ids = outputs.sequences
            sequences_scores = outputs.sequences_scores
            
            # 重塑结果以便按样本分组
            batch_size = encoder_input_ids.size(0)
            generated_ids_reshaped = generated_ids.view(batch_size, num_beams, -1)[:, :, 1:]
            probabilities_reshaped = torch.exp(sequences_scores).view(batch_size, num_beams)
            
            # 处理每个样本
            for i in range(batch_size):
                # 获取当前样本的真实物品ID
                true_item_id = true_item_ids[i]
                
                # 获取当前样本的所有生成序列和概率
                user_sequences = generated_ids_reshaped[i]
                user_probs = probabilities_reshaped[i]
                
                # 按概率从高到低排序
                sorted_indices = torch.argsort(user_probs, descending=True)
                sorted_sequences = user_sequences[sorted_indices]
                
                # 将tokens转换为item ID
                item_ids = []
                for seq in sorted_sequences:
                    item_id = tokens_to_item_id(seq, tokens_to_item_map)
                    item_ids.append(item_id)
                
                
                # 计算每个K值的指标
                for k in k_list:
                    # 取前k个推荐物品
                    top_k_items = item_ids[:k]
                    
                    # 计算Hit@K
                    hit = 1 if true_item_id in top_k_items else 0
                    metrics[f'hit@{k}'] += hit
                    
                    # 计算NDCG@K
                    if hit:
                        rank = top_k_items.index(true_item_id) + 1
                        ndcg = 1 / math.log2(rank + 1)
                    else:
                        ndcg = 0
                    metrics[f'ndcg@{k}'] += ndcg
                
                total_samples += 1
            
            # 更新进度条
            if accelerator.is_main_process and total_samples > 0:
                current_metrics = {}
                for k in k_list:
                    current_metrics[f"hit@{k}"] = metrics[f"hit@{k}"] / total_samples
                progress_bar.set_postfix(current_metrics)
    
    # 计算平均指标
    if total_samples > 0:
        for k in k_list:
            metrics[f'hit@{k}'] /= total_samples
            metrics[f'ndcg@{k}'] /= total_samples
    else:
        # 如果没有有效样本，设置默认值
        for k in k_list:
            metrics[f'hit@{k}'] = 0
            metrics[f'ndcg@{k}'] = 0
    
    # 打印结果
    if accelerator.is_main_process and logger:
        logger.info("\n测试结果:")
        for k in k_list:
            logger.info(f"Hit@{k}: {metrics[f'hit@{k}']:.4f}, NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        logger.info(f"总有效样本数: {total_samples}")
    
    return metrics




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
    
    # 初始化指标累计器
    metrics = {f'hit@{k}': 0 for k in k_list}
    metrics.update({f'ndcg@{k}': 0 for k in k_list})
    total_samples = 0
    
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
            # 处理每个样本
            for i in range(batch_size):
                # 获取当前样本的真实物品ID
                true_item_id = true_item_ids[i]
                
                # 获取当前样本的所有生成序列和概率
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
                
                # 计算每个K值的指标
                for k in k_list:
                    # 取前k个推荐物品
                    top_k_items = item_ids[:k]
                    
                    # 计算Hit@K
                    hit = 1 if true_item_id in top_k_items else 0
                    metrics[f'hit@{k}'] += hit
                    
                    # 计算NDCG@K
                    if hit:
                        rank = top_k_items.index(true_item_id) + 1
                        ndcg = 1 / math.log2(rank + 1)
                    else:
                        ndcg = 0
                    metrics[f'ndcg@{k}'] += ndcg
                
                total_samples += 1
            
            # 更新进度条
            if accelerator.is_main_process and total_samples > 0:
                current_metrics = {}
                for k in k_list:
                    current_metrics[f"hit@{k}"] = metrics[f"hit@{k}"] / total_samples
                progress_bar.set_postfix(current_metrics)
    metrics_tensor = torch.tensor([
        total_samples,
        *[metrics[f'hit@{k}'] for k in k_list],
        *[metrics[f'ndcg@{k}'] for k in k_list]
    ], dtype=torch.float32, device=device)
    
    gathered_metrics = accelerator.gather(metrics_tensor)
    
    if accelerator.is_main_process:
        # 添加调试信息
        logger.info(f"Original metrics_tensor shape: {metrics_tensor.shape}")
        logger.info(f"Gathered metrics shape: {gathered_metrics.shape}")
        logger.info(f"Gathered metrics dim: {gathered_metrics.dim()}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        
        # gather flatten了所有进程结果
        num_processes = accelerator.num_processes
        metrics_per_process = len(metrics_tensor)
        gathered_metrics = gathered_metrics.view(num_processes, metrics_per_process)
        
        total_gathered_samples = gathered_metrics[:, 0].sum().item()
        
        aggregated_metrics = {}
        idx = 1
        for k in k_list:
            aggregated_metrics[f'hit@{k}'] = gathered_metrics[:, idx].sum().item() / total_gathered_samples if total_gathered_samples > 0 else 0
            idx += 1
        for k in k_list:
            aggregated_metrics[f'ndcg@{k}'] = gathered_metrics[:, idx].sum().item() / total_gathered_samples if total_gathered_samples > 0 else 0
            idx += 1
        
        if logger:
            logger.info(f"\n{mode}结果:")
            for k in k_list:
                logger.info(f"Hit@{k}: {aggregated_metrics[f'hit@{k}']:.4f}, NDCG@{k}: {aggregated_metrics[f'ndcg@{k}']:.4f}")
            logger.info(f"总有效样本数: {int(total_gathered_samples)}")
        
        return aggregated_metrics
    else:
        return {f'hit@{k}': 0 for k in k_list} | {f'ndcg@{k}': 0 for k in k_list}
    
    # # 计算平均指标
    # if total_samples > 0:
    #     for k in k_list:
    #         metrics[f'hit@{k}'] /= total_samples
    #         metrics[f'ndcg@{k}'] /= total_samples
    # else:
    #     # 如果没有有效样本，设置默认值
    #     for k in k_list:
    #         metrics[f'hit@{k}'] = 0
    #         metrics[f'ndcg@{k}'] = 0
    
    # # 打印结果
    # if accelerator.is_main_process and logger:
    #     logger.info(f"\n{mode}结果:")
    #     for k in k_list:
    #         logger.info(f"Hit@{k}: {metrics[f'hit@{k}']:.4f}, NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    #     logger.info(f"总有效样本数: {total_samples}")
    
    # return metrics
