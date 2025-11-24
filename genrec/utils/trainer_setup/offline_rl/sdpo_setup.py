from typing import Optional, Dict, List  
from functools import partial  
from transformers import TrainingArguments, EarlyStoppingCallback  
from genrec.utils.metrics import compute_metrics  
from genrec.trainers.offline_rl.sdpo_trainer import SDPOTrainer  
from genrec.utils.callbacks.generative.generative_callback import (  
    GenerativeLoggingCallback,  
    EvaluateEveryNEpochsCallback  
)  
from genrec.utils.models_setup.conditional_t5_setup import create_t5_model  

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
    eval_data_collator = None,  
    # 生成评估参数  
    compute_metrics: Optional[callable] = None,  
    generation_params: Optional[Dict] = None,  
    item2tokens: Optional[Dict] = None,  
    pad_token_id: Optional[int] = None,  
    eos_token_id: Optional[int] = None,  
    **kwargs  
):  
    """  
    创建 SDPOTrainer 的工厂函数  
      
    Args:  
        model: 策略模型
        training_args: 训练参数
        train_dataset: 训练数据集（包含 chosen/rejected labels）
        eval_dataset: 评估数据集（只需要 chosen_labels）
        data_collator: 训练数据 collator
        callbacks: 回调函数列表
        ref_model: 参考模型（用于 S-DPO）
        beta: S-DPO 温度参数
        eval_data_collator: 评估数据 collator
        compute_metrics: 评估指标计算函数
        generation_params: 生成参数（max_gen_length, num_beams, max_k）
        item2tokens: item 到 token 的映射（用于前缀约束）
        pad_token_id: pad token id
        eos_token_id: eos token id
    """  
      
    # 检查必需参数  
    if ref_model is None:  
        raise ValueError("使用 SDPOTrainer 时需要提供 ref_model 参数")  
    
    if None in [compute_metrics, generation_params, item2tokens, pad_token_id, eos_token_id]:
        raise ValueError("使用 SDPOTrainer 进行生成评估时需要提供 compute_metrics, "
                       "generation_params, item2tokens, pad_token_id 和 eos_token_id 参数")
      
    return SDPOTrainer(  
        model=model,  
        ref_model=ref_model,  
        beta=beta,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=eval_dataset,  
        data_collator=data_collator,  
        eval_data_collator=eval_data_collator,  
        callbacks=callbacks,  
        compute_metrics=compute_metrics,  
        generation_params=generation_params,  
        item2tokens=item2tokens,  
        pad_token_id=pad_token_id,  
        eos_token_id=eos_token_id,  
        **kwargs  
    )  

def setup_training(  
    model,   
    tokenizer,   
    train_dataset,   
    valid_dataset,   
    model_config,   
    offline_rl_config,  
    output_dirs,   
    logger,   
    per_device_train_batch_size,  
    per_device_eval_batch_size,   
    train_data_collator,  
    eval_data_collator  
):  
    """
    设置 S-DPO 训练
    
    Args:
        model: 策略模型
        tokenizer: 分词器
        train_dataset: 训练数据集（包含 chosen/rejected labels）
        valid_dataset: 验证数据集（只需要 chosen_labels，用于生成评估）
        model_config: 模型配置
        offline_rl_config: 离线强化学习配置
        output_dirs: 输出目录
        logger: 日志记录器
        per_device_train_batch_size: 训练批次大小
        per_device_eval_batch_size: 评估批次大小
        train_data_collator: 训练数据 collator（处理 chosen/rejected）
        eval_data_collator: 评估数据 collator（只处理 chosen）
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
    
    # ===== 5. 创建 Trainer =====
    trainer = create_trainer(  
        model=model,  
        training_args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=valid_dataset,  
        data_collator=train_data_collator,  
        eval_data_collator=eval_data_collator,  
        callbacks=callbacks,  
        # S-DPO 参数  
        ref_model=ref_model,  
        beta=offline_rl_config.get('beta', 0.1),  
        # 生成评估参数  
        compute_metrics=compute_metrics_with_map,  
        generation_params=generation_params,  
        item2tokens=tokenizer.item2tokens,  
        pad_token_id=tokenizer.pad_token,  
        eos_token_id=tokenizer.eos_token,  
    )  
    
    logger.info(f"Trainer 配置完成:")
    logger.info(f"  - Beta: {offline_rl_config.get('beta', 0.1)}")
    logger.info(f"  - Num beams: {num_beams}")
    logger.info(f"  - Max gen length: {max_gen_length}")
    logger.info(f"  - Max k: {max_k}")
    logger.info(f"  - Metric for best model: {training_args.metric_for_best_model}")
      
    return trainer