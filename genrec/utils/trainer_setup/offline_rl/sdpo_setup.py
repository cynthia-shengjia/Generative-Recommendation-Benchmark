from typing import Optional, Dict, List
from functools import partial
from transformers import TrainingArguments, EarlyStoppingCallback
from transformers import TrainingArguments
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
    # 自定义Trainer特有参数
    compute_metrics: Optional[callable] = None,
    generation_params: Optional[Dict] = None,
    item2tokens: Optional[Dict] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    ref_model: Optional = None,  # 新增：S-DPO 需要
    beta: float = 0.1,
    eval_data_collator = None,
    **kwargs
):
    """
    创建Trainer的工厂函数
    
    Args:
        use_generative_trainer: 是否使用自定义的GenerativeTrainer
        其他参数与标准Trainer参数一致
        generation_params: 生成参数，仅当use_generative_trainer=True时使用
        item2tokens: item到token的映射，仅当use_generative_trainer=True时使用
        pad_token_id: pad token id，仅当use_generative_trainer=True时使用
        eos_token_id: eos token id，仅当use_generative_trainer=True时使用
    """
    
    if ref_model is None:
        raise ValueError("使用SDPOTrainer时需要提供ref_model参数")
    
    return SDPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=beta,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        eval_data_collator = eval_data_collator,
        **kwargs
    )

# 使用示例
def setup_training(
    model, 
    tokenizer, 
    train_dataset, 
    valid_dataset, 
    model_config, 
    output_dirs, 
    logger, 
    per_device_train_batch_size,
    per_device_eval_batch_size, 
    train_data_collator,
    eval_data_collator
):

    # 公共TrainingArguments配置
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
    )
    
    # 根据选择的Trainer类型设置不同的参数
         # S-DPO 特定配置



    training_args.metric_for_best_model = "loss"
    training_args.greater_is_better = True
    
    # 准备GenerativeTrainer特有参数
    tokens_to_item_map = tokenizer.tokens2item
    compute_metrics_with_map = partial(compute_metrics, tokens_to_item_map=tokens_to_item_map)
    
    num_beams = model_config.get('num_beams', 10)
    max_gen_length = model_config.get('max_gen_length', 5)
    k_list = model_config.get('k_list', [])
    max_k = k_list[-1]
    
    generation_params = {
        'max_gen_length': max_gen_length,
        'num_beams': num_beams,
        'max_k': max_k
    }
    
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=model_config.get("early_stop_upper_steps", 1000)), 
        GenerativeLoggingCallback(logger), 
        EvaluateEveryNEpochsCallback(n_epochs=model_config.get("evaluation_epoch", 5))
    ]
    
    
    # 创建 reference model
    ref_model = create_t5_model(
        vocab_size=tokenizer.vocab_size,
        model_config=model_config
    )
    # 加载与 model 相同的权重
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=model_config.get("early_stop_upper_steps", 1000)),
        GenerativeLoggingCallback(logger),
    ]
    
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=train_data_collator,
        callbacks=callbacks,
        ref_model=ref_model,
        beta=model_config.get('beta', 0.1),
        compute_metrics=compute_metrics_with_map,
        generation_params=generation_params,
        item2tokens=tokenizer.item2tokens,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        eval_data_collator=eval_data_collator,  # ✅ 传入 eval collator
    )    

    
    return trainer
