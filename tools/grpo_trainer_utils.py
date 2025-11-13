# tools/grpo_trainer_utils.py

import os
import torch
from typing import Dict, List, Optional
from trl import GRPOConfig
from transformers import TrainerCallback, TrainingArguments, TrainerState
from trainers.rl_trainers.GRPOTrainer import GRPOTrainerForGenRec
from functools import partial
import logging

class GRPOLoggingCallback(TrainerCallback):
    """
    GRPO 专用的日志回调函数
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            if any(key.startswith("eval_") for key in logs.keys()):
                self.logger.info("***** GRPO 验证结果 *****")
                for key, value in logs.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                # 过滤并格式化日志
                _logs = {k: v for k, v in logs.items() if k not in ["epoch", "step"]}
                
                # GRPO 特有的指标
                grpo_metrics = ["reward", "reward_std", "kl", "accuracy", "diversity", "completion_length"]
                grpo_log_items = []
                other_log_items = []
                
                for k, v in _logs.items():
                    formatted = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    if any(metric in k for metric in grpo_metrics):
                        grpo_log_items.append(formatted)
                    else:
                        other_log_items.append(formatted)
                
                log_str = f"步骤 {state.global_step} (Epoch {state.epoch:.2f})"
                if other_log_items:
                    log_str += " | " + " | ".join(other_log_items)
                if grpo_log_items:
                    log_str += "\n  GRPO指标: " + " | ".join(grpo_log_items)
                
                self.logger.info(log_str)

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