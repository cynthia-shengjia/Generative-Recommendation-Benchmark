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
from typing import Optional, Dict, Any, List, Union, Callable  # 添加了必要的导入
import math
import logging
import re
from transformers import T5ForConditionalGeneration,T5Config,Trainer
import numpy as np
import math
from transformers import EvalPrediction
from functools import partial
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import TrainerCallback, TrainingArguments, TrainerState
from trainers.model_trainers.MLSTrainer import MSLTrainer
from tools.nni_utils import get_nni_params, update_config_with_nni, report_nni_metrics


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



# 自定义回调函数来控制评估频率
class EvaluateEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs
        self.last_eval_epoch = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # 每隔n_epochs开启评估，否则关闭
        if (state.epoch ) % self.n_epochs == 0:
            control.should_evaluate = True
            self.last_eval_epoch = state.epoch
        else:
            control.should_evaluate = False
            
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 仅在评估时保存检查点
        control.should_save = state.epoch == self.last_eval_epoch


class LoggingCallback(TrainerCallback):
    """
    一个自定义的回调函数，将 Trainer 的日志（包括训练进度和评估结果）
    转发到指定的 logger。
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            if any(key.startswith("eval_") for key in logs.keys()):
                self.logger.info("***** 验证结果 *****")
                metrics = {}
                for key, value in logs.items():
                    self.logger.info(f"  {key}: {value}")
                    metrics.update({key: value})
                if "NNI_PLATFORM" in os.environ:
                    is_final = state.epoch >= args.num_train_epochs
                    report_nni_metrics(metrics,is_final)
            else: 
                _logs = {k: v for k, v in logs.items() if k not in ["epoch", "step"]}
                log_str = f"步骤 {state.global_step} (Epoch {state.epoch:.2f}): " + " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in _logs.items())
                self.logger.info(log_str)



def create_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    data_collator,
    # 通用参数
    callbacks: Optional[List] = None,
    # 自定义Trainer特有参数
    use_generative_trainer: bool = False,
    compute_metrics: Optional[callable] = None,
    generation_params: Optional[Dict] = None,
    item2tokens: Optional[Dict] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    tau: float = None,
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
    
    if use_generative_trainer:
        # 检查必需参数
        if None in [compute_metrics, generation_params, item2tokens, pad_token_id, eos_token_id]:
            raise ValueError("使用GenerativeTrainer时需要提供compute_metrics, generation_params, "
                           "item2tokens, pad_token_id和eos_token_id参数")
        
        # 创建自定义Trainer
        return MSLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            generation_params=generation_params,
            item2tokens=item2tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tau=tau,
            **kwargs
        )
    else:
        # 创建标准Trainer
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            **kwargs
        )

# 使用示例
def setup_training(model, tokenizer, train_dataset, valid_dataset, model_config, output_dirs, train_data_collator, logger, use_generative=False):
    # 公共TrainingArguments配置
    training_args = TrainingArguments(
        output_dir=output_dirs['model'],
        num_train_epochs=model_config['num_epochs'],
        per_device_train_batch_size=model_config['batch_size'],
        per_device_eval_batch_size=model_config['test_batch_size'],
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=output_dirs['logs'],
        logging_steps=100,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    
    # 根据选择的Trainer类型设置不同的参数
    if use_generative:
        training_args.metric_for_best_model = "ndcg@10"
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
            LoggingCallback(logger), 
            EvaluateEveryNEpochsCallback(n_epochs=model_config.get("evaluation_epoch", 5))
        ]
    else:
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
        callbacks = [EarlyStoppingCallback(early_stopping_patience=model_config.get("early_stop_upper_steps", 1000))]
        compute_metrics_with_map = None
        generation_params = None
    
    # 创建Trainer
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=train_data_collator,
        callbacks=callbacks,
        use_generative_trainer=use_generative,
        compute_metrics=compute_metrics_with_map,
        generation_params=generation_params,
        item2tokens=tokenizer.item2tokens if use_generative else None,
        pad_token_id=tokenizer.pad_token if use_generative else None,
        eos_token_id=tokenizer.eos_token if use_generative else None,
        tau=model_config.get("temperature", 1)
    )
    
    return trainer
