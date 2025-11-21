# tools/grpo_trainer_utils.py

import os
import torch
from typing import Dict, List, Optional
from trl import GRPOConfig
from trainers.rl_trainers.GRPOTrainer import GRPOTrainerForGenRec
from functools import partial
import logging
import math


def create_grpo_reward_function(tokenizer):
    """
    创建 GRPO 的奖励函数
    
    Args:
        tokenizer: TigerTokenizer 实例
    
    Returns:
        reward_func: 奖励函数
    """
    def reward_func(generated_items: List[int], target_items: List[int]) -> List[float]:
        """
        奖励函数：生成的物品与目标物品匹配则给予 1.0，否则 0.0
        
        Args:
            generated_items: 生成的物品 ID 列表
            target_items: 目标物品 ID 列表
        
        Returns:
            rewards: 奖励列表
        """
        rewards = []
        for gen_item, target_item in zip(generated_items, target_items):
            if gen_item == target_item:
                rewards.append(1.0)
            else:
                # 可以根据需要添加更复杂的奖励逻辑
                # 例如：部分匹配、流行度惩罚等
                rewards.append(0.0)
        return rewards


    return reward_func



# def create_grpo_reward_function(tokenizer, use_ndcg=True, ndcg_weight=0.5):   
#     """   
#     创建 GRPO 的奖励函数  
      
#     Args:   
#         tokenizer: TigerTokenizer 实例  
#         use_ndcg: 是否使用 NDCG 奖励
#         ndcg_weight: NDCG 奖励的权重 (0-1之间)
      
#     Returns:   
#         reward_func: 奖励函数  
#     """   
#     def reward_func(generated_items: List[int], target_items: List[int], 
#                    num_generations: int) -> List[float]:   
#         """   
#         奖励函数：结合匹配奖励和 NDCG 奖励
#         注意：generated_items 已经按照 beam search 的分数排序（从高到低）
          
#         Args:   
#             generated_items: 生成的物品 ID 列表 [B * num_generations]
#             target_items: 目标物品 ID 列表 [B * num_generations]
#             num_generations: 每个样本的生成数量
          
#         Returns:   
#             rewards: 奖励列表  
#         """   
#         # 预计算 NDCG 负奖励（只需计算一次）
#         ndcg_penalties = [-1.0/math.log2(i+2) for i in range(num_generations)]
#         ndcg_sum = sum(ndcg_penalties)
#         ndcg_penalties = [-elm/ndcg_sum for elm in ndcg_penalties]
        
#         rewards = []
        
#         # 按组处理（每组有 num_generations 个生成结果）
#         for group_idx in range(len(generated_items) // num_generations):
#             start_idx = group_idx * num_generations
#             end_idx = start_idx + num_generations
            
#             # 获取当前组的数据
#             group_gen_items = generated_items[start_idx:end_idx]
#             group_target_items = target_items[start_idx:end_idx]
            
#             # 注意：group_gen_items 已经按照概率从高到低排序
#             # rank 0 是概率最高的，rank num_generations-1 是概率最低的
#             for rank, (gen_item, target_item) in enumerate(zip(group_gen_items, group_target_items)):
#                 # 基础匹配奖励
#                 match_reward = 1.0 if gen_item == target_item else 0.0
                
#                 if not use_ndcg:
#                     # 不使用 NDCG，只用匹配奖励
#                     final_reward = match_reward
#                 else:
#                     if match_reward == 1.0:  # 正样本
#                         # 正样本的 NDCG 奖励为 0
#                         final_reward = (1 - ndcg_weight) * match_reward + ndcg_weight * 0.0
#                     else:  # 负样本
#                         # 负样本根据排名获得负奖励
#                         # rank 越大（排名越靠后），惩罚越小（绝对值）
#                         final_reward = (1 - ndcg_weight) * match_reward + ndcg_weight * ndcg_penalties[rank]
                
#                 rewards.append(final_reward)
#         return rewards

      
#     return reward_func



def setup_grpo_training(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    grpo_config: dict,
    output_dirs: dict,
    logger,
    pretrained_model_path: Optional[str] = None
):
    """
    设置 GRPO 训练
    
    Args:
        model: T5 模型
        tokenizer: TigerTokenizer
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        grpo_config: GRPO 配置字典
        output_dirs: 输出目录字典
        logger: 日志记录器
        pretrained_model_path: 预训练模型路径（可选）
    
    Returns:
        trainer: GRPOTrainerForGenRec 实例
    """
    
    # 如果提供了预训练模型路径，加载模型
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        logger.info(f"从 {pretrained_model_path} 加载预训练模型...")
        model.load_state_dict(torch.load(pretrained_model_path))
        logger.info("预训练模型加载完成")
    
    # 创建 GRPO 配置
    training_args = GRPOConfig(
        output_dir=output_dirs['grpo_model'],
        num_train_epochs=grpo_config.get('num_epochs', 3),
        per_device_train_batch_size=grpo_config.get('batch_size', 4),
        per_device_eval_batch_size=grpo_config.get('test_batch_size', 8),
        learning_rate=grpo_config.get('learning_rate', 1e-5),
        weight_decay=grpo_config.get('weight_decay', 0.01),
        
        # GRPO 特有参数
        max_prompt_length=grpo_config.get('max_prompt_length', 512),
        max_completion_length=grpo_config.get('max_completion_length', 5),
        num_generations=grpo_config.get('num_generations', 4),
        beta=grpo_config.get('beta', 0.1),  # KL 惩罚系数
        temperature=grpo_config.get('temperature', 1.0),
        
        # 评估和保存策略
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # 使用 eval_loss 作为最佳模型指标
        greater_is_better=True,
        
        # 日志
        logging_dir=output_dirs['logs'],
        logging_steps=grpo_config.get('logging_steps', 10),
        report_to=[],
        
        # 其他
        seed=grpo_config.get('seed', 42),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    
    # 创建奖励函数
    reward_func = create_grpo_reward_function(tokenizer)
    # reward_func = create_grpo_reward_function(tokenizer)

    
    # 创建回调函数
    callbacks = [
        GRPOLoggingCallback(logger)
    ]
    
    # 创建 data collator
    from genrec.datasets.data_collator import TrainSeqRecDataCollator
    tokens_per_item = len(next(iter(tokenizer.item2tokens.values())))
    data_collator = TrainSeqRecDataCollator(
        max_seq_len=train_dataset.max_token_len,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        tokens_per_item=tokens_per_item
    )
    
    # 创建 GRPO Trainer
    trainer = GRPOTrainerForGenRec(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        reward_func=reward_func,
        callbacks=callbacks,
    )
    
    logger.info("GRPO Trainer 创建完成")
    logger.info(f"  训练样本数: {len(train_dataset)}")
    logger.info(f"  验证样本数: {len(eval_dataset)}")
    logger.info(f"  每设备批次大小: {training_args.per_device_train_batch_size}")
    logger.info(f"  每个 prompt 生成数: {training_args.num_generations}")
    logger.info(f"  KL 惩罚系数 (beta): {training_args.beta}")
    logger.info(f"  学习率: {training_args.learning_rate}")
    
    return trainer