# genrec/utils/trainer_setup/offline_rl/offline_rl_setup.py

from typing import Optional, Dict, List
from functools import partial
from transformers import TrainingArguments, EarlyStoppingCallback
from hydra.utils import instantiate
from omegaconf import DictConfig

from genrec.utils.metrics import compute_metrics
from genrec.utils.callbacks.generative.generative_callback import (
    GenerativeLoggingCallback,
    EvaluateEveryNEpochsCallback
)
from genrec.utils.models_setup.tiger_setup import create_tiger_model

def setup_training(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    model_config,
    offline_rl_config: DictConfig,
    output_dirs,
    logger,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    train_data_collator,
    eval_data_collator,
):


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
    

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=model_config.get("early_stop_upper_steps", 1000)
        ),
        GenerativeLoggingCallback(logger),
        EvaluateEveryNEpochsCallback(
            n_epochs=model_config.get("evaluation_epoch", 5)
        )
    ]
    

    ref_model = create_tiger_model(
        vocab_size=tokenizer.vocab_size,
        model_config=model_config
    )
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    trainer_partial = instantiate(offline_rl_config.trainer)

    trainer = trainer_partial(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=train_data_collator,
        eval_data_collator=eval_data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics_with_map,
        generation_params=generation_params,
        item2tokens=tokenizer.item2tokens,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
    )
    
    return trainer