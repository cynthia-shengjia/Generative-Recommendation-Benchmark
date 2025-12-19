# genrec/utils/trainer_setup/online_rl/online_rl_setup.py

from typing import Optional, Dict, List, Callable
from functools import partial
from transformers import TrainingArguments, EarlyStoppingCallback
from hydra.utils import instantiate
from omegaconf import DictConfig

from genrec.utils.metrics import compute_metrics
from genrec.utils.callbacks.generative.generative_callback import (
    GenerativeLoggingCallback,
    EvaluateEveryNEpochsCallback
)
from genrec.utils.models_setup.conditional_t5_setup import create_t5_model

import math

def create_grpo_reward_function(use_ndcg=True, ndcg_weight=0.5):
    """创建 GRPO 的奖励函数"""
    def reward_func(generated_items: List[int], target_items: List[int],
                   num_generations: int) -> List[float]:
        ndcg_penalties = [-1.0/math.log2(i+2) for i in range(num_generations)]
        ndcg_sum = sum(ndcg_penalties)
        ndcg_penalties = [-elm/ndcg_sum for elm in ndcg_penalties]
        
        rewards = []
        for group_idx in range(len(generated_items) // num_generations):
            start_idx = group_idx * num_generations
            end_idx = start_idx + num_generations
            
            group_gen_items = generated_items[start_idx:end_idx]
            group_target_items = target_items[start_idx:end_idx]
            
            for rank, (gen_item, target_item) in enumerate(zip(group_gen_items, group_target_items)):
                match_reward = 1.0 if gen_item == target_item else 0.0
                
                if not use_ndcg:
                    final_reward = match_reward
                else:
                    if match_reward == 1.0:
                        final_reward = (1 - ndcg_weight) * match_reward + ndcg_weight * 0.0
                    else:
                        final_reward = (1 - ndcg_weight) * match_reward + ndcg_weight * ndcg_penalties[rank]
                
                rewards.append(final_reward)
        return rewards
    
    return reward_func

def setup_training(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    model_config,
    online_rl_config: DictConfig,
    output_dirs,
    logger,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    train_data_collator,
):
    """
    统一的 Online RL 训练设置函数
    """
    
    # ===== 1. 训练参数配置 =====
    training_args = TrainingArguments(
        output_dir=output_dirs['model'],
        num_train_epochs=model_config['num_epochs'],
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=output_dirs['logs'],
        logging_steps=100,
        report_to=[],
        warmup_ratio=model_config["warmup_ratio"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        metric_for_best_model="ndcg@10",
        greater_is_better=True,
    )
    
    # ===== 2. 生成评估参数 =====
    tokens_to_item_map = tokenizer.tokens2item
    compute_metrics_with_map = partial(
        compute_metrics,
        tokens_to_item_map=tokens_to_item_map
    )
    
    num_beams = model_config.get('num_beams', 10)
    max_gen_length = model_config.get('max_gen_length', 5)
    k_list = model_config.get('k_list', [5, 10, 20])
    max_k = k_list[-1] if k_list else 10
    
    generation_params = {
        'max_gen_length': max_gen_length,
        'num_beams': num_beams,
        'max_k': max_k
    }
    
    # ===== 3. 回调函数 =====
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=model_config.get("early_stop_upper_steps", 1000)
        ),
        GenerativeLoggingCallback(logger),
        EvaluateEveryNEpochsCallback(
            n_epochs=model_config.get("evaluation_epoch", 5)
        )
    ]
    
    # ===== 4. 创建参考模型 =====
    logger.info("创建参考模型（Reference Model）...")
    ref_model = create_t5_model(
        vocab_size=tokenizer.vocab_size,
        model_config=model_config
    )
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("参考模型创建完成")
    
    # ===== 5. 创建奖励函数（如果需要）=====
    reward_func = None
    if 'reward_func' in online_rl_config.trainer:
        reward_func = instantiate(online_rl_config.trainer.reward_func)
    
    # ===== 6. 使用 partial instantiate 创建 Trainer =====
    logger.info(f"实例化 Trainer: {online_rl_config.trainer._target_}")
    
    # 🔥 使用 instantiate 获取 partial 函数
    trainer_partial = instantiate(online_rl_config.trainer)
    
    # 🔥 调用 partial 函数，传入运行时参数
    trainer = trainer_partial(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=train_data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics_with_map,
        generation_params=generation_params,
        item2tokens=tokenizer.item2tokens,
        tokens2item=tokenizer.tokens2item,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        reward_func=reward_func,
    )
    
    logger.info(f"Trainer 配置完成:")
    logger.info(f"  - Trainer 类型: {online_rl_config.trainer._target_}")
    logger.info(f"  - Beta: {online_rl_config.trainer.get('beta', 'N/A')}")
    logger.info(f"  - Num generations: {online_rl_config.trainer.get('num_generations', 'N/A')}")
    logger.info(f"  - Num beams: {num_beams}")
    logger.info(f"  - Max gen length: {max_gen_length}")
    logger.info(f"  - Max k: {max_k}")
    
    return trainer