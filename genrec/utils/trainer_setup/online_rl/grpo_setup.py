from typing import Optional, Dict, List, Callable
from functools import partial  
from transformers import TrainingArguments, EarlyStoppingCallback  
from genrec.utils.metrics import compute_metrics  
from genrec.trainers.online_rl.grpo_trainer import GRPOTrainer  
from genrec.utils.callbacks.generative.generative_callback import (  
    GenerativeLoggingCallback,  
    EvaluateEveryNEpochsCallback  
)  
from genrec.utils.models_setup.conditional_t5_setup import create_t5_model  

import math


def create_grpo_reward_function(use_ndcg=True, ndcg_weight=0.5):   
    """   
    创建 GRPO 的奖励函数  
      
    Args:   
        tokenizer: TigerTokenizer 实例  
        use_ndcg: 是否使用 NDCG 奖励
        ndcg_weight: NDCG 奖励的权重 (0-1之间)
      
    Returns:   
        reward_func: 奖励函数  
    """   
    def reward_func(generated_items: List[int], target_items: List[int], 
                   num_generations: int) -> List[float]:   
        """   
        奖励函数：结合匹配奖励和 NDCG 奖励
        注意：generated_items 已经按照 beam search 的分数排序（从高到低）
          
        Args:   
            generated_items: 生成的物品 ID 列表 [B * num_generations]
            target_items: 目标物品 ID 列表 [B * num_generations]
            num_generations: 每个样本的生成数量
          
        Returns:   
            rewards: 奖励列表  
        """   
        # 预计算 NDCG 负奖励（只需计算一次）
        ndcg_penalties = [-1.0/math.log2(i+2) for i in range(num_generations)]
        ndcg_sum = sum(ndcg_penalties)
        ndcg_penalties = [-elm/ndcg_sum for elm in ndcg_penalties]
        
        rewards = []
        
        # 按组处理（每组有 num_generations 个生成结果）
        for group_idx in range(len(generated_items) // num_generations):
            start_idx = group_idx * num_generations
            end_idx = start_idx + num_generations
            
            # 获取当前组的数据
            group_gen_items = generated_items[start_idx:end_idx]
            group_target_items = target_items[start_idx:end_idx]
            
            # 注意：group_gen_items 已经按照概率从高到低排序
            # rank 0 是概率最高的，rank num_generations-1 是概率最低的
            for rank, (gen_item, target_item) in enumerate(zip(group_gen_items, group_target_items)):
                # 基础匹配奖励
                match_reward = 1.0 if gen_item == target_item else 0.0
                
                if not use_ndcg:
                    # 不使用 NDCG，只用匹配奖励
                    final_reward = match_reward
                else:
                    if match_reward == 1.0:  # 正样本
                        # 正样本的 NDCG 奖励为 0
                        final_reward = (1 - ndcg_weight) * match_reward + ndcg_weight * 0.0
                    else:  # 负样本
                        # 负样本根据排名获得负奖励
                        # rank 越大（排名越靠后），惩罚越小（绝对值）
                        final_reward = (1 - ndcg_weight) * match_reward + ndcg_weight * ndcg_penalties[rank]
                
                rewards.append(final_reward)
        return rewards

      
    return reward_func

def create_trainer(  
    model,  
    training_args,  
    train_dataset,  
    eval_dataset,  
    data_collator,
    # 通用参数  
    callbacks: Optional[List] = None,  
    # S-DPO 特有参数  
    ref_model: Optional = None,  
    beta: float = 0.1,  
    num_generations: int = 2,
    # 生成评估参数  
    compute_metrics: Optional[callable] = None,  
    generation_params: Optional[Dict] = None,  
    item2tokens: Optional[Dict] = None,  
    tokens2item: Optional[Dict] = None,  
    pad_token_id: Optional[int] = None,  
    eos_token_id: Optional[int] = None,  
    reward_func: Optional[Callable] = None,
    **kwargs  
):  
    """  
    创建 GRPOTrainer 的工厂函数  
      
    Args:  
        model: 策略模型
        training_args: 训练参数
        train_dataset: 训练数据集（包含 chosen/rejected labels）
        eval_dataset: 评估数据集（只需要 chosen_labels）
        data_collator: 训练数据 collator
        callbacks: 回调函数列表
        ref_model: 参考模型（用于 S-DPO）
        beta: S-DPO 温度参数
        compute_metrics: 评估指标计算函数
        generation_params: 生成参数（max_gen_length, num_beams, max_k）
        item2tokens: item 到 token 的映射（用于前缀约束）
        pad_token_id: pad token id
        eos_token_id: eos token id
    """  
      
    # 检查必需参数  
    if ref_model is None:  
        raise ValueError("使用 GRPOTrainer 时需要提供 ref_model 参数")  
    
    if None in [compute_metrics, generation_params, item2tokens, pad_token_id, eos_token_id]:
        raise ValueError("使用 GRPOTrainer 进行生成评估时需要提供 compute_metrics, "
                       "generation_params, item2tokens, pad_token_id 和 eos_token_id 参数")
      
    return GRPOTrainer(  
        model=model,  
        ref_model = ref_model,
        beta=beta,  
        num_generations = num_generations,
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=eval_dataset,  
        data_collator=data_collator,  
        callbacks=callbacks,  
        # 🔴 生成评估参数  
        compute_metrics=compute_metrics,  
        generation_params=generation_params,  
        item2tokens=item2tokens,  
        tokens2item=tokens2item,
        pad_token_id=pad_token_id,  
        eos_token_id=eos_token_id,  
        reward_func = reward_func,
        **kwargs  
    )  

def setup_training(  
    model,   
    tokenizer,   
    train_dataset,   
    valid_dataset,   
    model_config,   
    online_rl_config,  
    output_dirs,   
    logger,   
    per_device_train_batch_size,  
    per_device_eval_batch_size,   
    train_data_collator,  
):  
    """
    设置 GRPO 训练
    
    Args:
        model: 策略模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        valid_dataset: 验证数据集
        model_config: 模型配置
        online_rl_config: 离线强化学习配置
        output_dirs: 输出目录
        logger: 日志记录器
        per_device_train_batch_size: 训练批次大小
        per_device_eval_batch_size: 评估批次大小
        train_data_collator: 训练数据 collator（处理 chosen/rejected）
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
        # 🔴 评估指标配置（使用生成评估指标）
        metric_for_best_model="ndcg@10",  # 或 "recall@10"
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
    # 🔴 加载与策略模型相同的权重  
    ref_model.load_state_dict(model.state_dict())  
    ref_model.eval()  # 设置为评估模式
    # 🔴 冻结参考模型参数
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("参考模型创建完成")
    


    reward_func = create_grpo_reward_function(use_ndcg=True, ndcg_weight=0.5)

    # ===== 5. 创建 Trainer =====
    trainer = create_trainer(  
        model=model,  
        training_args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=valid_dataset,  
        data_collator=train_data_collator,  
        callbacks=callbacks,  
        # S-DPO 参数  
        ref_model=ref_model,  
        beta=online_rl_config.get('beta', 0.1),  
        num_generations = online_rl_config.get("num_generations",2),
        # 生成评估参数  
        compute_metrics=compute_metrics_with_map,  
        generation_params=generation_params,  
        item2tokens=tokenizer.item2tokens,  
        tokens2item=tokenizer.tokens2item,
        pad_token_id=tokenizer.pad_token,  
        eos_token_id=tokenizer.eos_token,  
        reward_func=reward_func
    )  
    
    logger.info(f"Trainer 配置完成:")
    logger.info(f"  - Beta: {online_rl_config.get('beta', 0.1)}")
    logger.info(f"  - Num beams: {num_beams}")
    logger.info(f"  - Max gen length: {max_gen_length}")
    logger.info(f"  - Max k: {max_k}")
    logger.info(f"  - Metric for best model: {training_args.metric_for_best_model}")
      
    return trainer