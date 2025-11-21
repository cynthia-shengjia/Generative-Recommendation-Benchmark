import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments
)

import pandas as pd
import numpy as np
import pickle
import os
import time
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any

from disrec.sasrec.sasrec4hf import SASRecConfig,SASRec4HF
from disrec.datasets.model_dataset import SASRecDataset
from disrec.datasets.data_collator import SASRecDataCollator

from transformers import TrainerCallback,EarlyStoppingCallback
import numpy as np
class EvaluateEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs
        self.last_eval_epoch = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch ) % self.n_epochs == 0:
            control.should_evaluate = True
            self.last_eval_epoch = state.epoch
        else:
            control.should_evaluate = False
            
    def on_evaluate(self, args, state, control, metrics, **kwargs):

        control.should_save = state.epoch == self.last_eval_epoch
def compute_metrics(eval_pred):

    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    if isinstance(logits, tuple):
        logits = logits[0]

    valid_mask = labels != -100 

    has_valid_label = valid_mask.any(axis=1)

    valid_labels = labels[has_valid_label]
    valid_logits = logits[has_valid_label]
    valid_mask = valid_mask[has_valid_label]

    last_item_indices = valid_mask.shape[1] - 1 - np.argmax(np.fliplr(valid_mask), axis=1)

    batch_indices = np.arange(valid_labels.shape[0])
    true_labels = valid_labels[batch_indices, last_item_indices]
    pred_logits = valid_logits

    K_VALUES = [1, 5, 10]
    sorted_indices = np.argsort(pred_logits, axis=1)[:, ::-1] 
    
    results = {}
    
    true_labels_reshaped = true_labels[:, np.newaxis]
    
    for k in K_VALUES:
        top_k_preds = sorted_indices[:, :k]

        hits = (true_labels_reshaped == top_k_preds).any(axis=1)
        hr_at_k = np.mean(hits)
        results[f"Hit@{k}"] = hr_at_k

        hit_mask = (true_labels_reshaped == top_k_preds)

        hit_positions = np.where(hit_mask) 

        ranks = hit_positions[1] 

        dcg = 1.0 / np.log2(ranks + 2)
        ndcg_at_k = np.sum(dcg) / len(true_labels)
        results[f"NDCG@{k}"] = ndcg_at_k

    return results
def load_data_and_get_vocab_size(file_path: str) -> int:

    print(f"正在从文件加载数据: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 数据文件未找到，请检查路径: {file_path}")
    
    df = pd.read_pickle(file_path)

    

    if 'ItemID' not in df.columns or df['ItemID'].apply(len).sum() == 0:
        raise ValueError("错误: 'ItemID' 列不存在或所有序列都为空，无法确定词汇表大小。")
        

    max_item_id = df['ItemID'].explode().max()
    

    vocab_size = int(max_item_id) + 1
    
    print(f"从数据中找到的最大 ItemID: {max_item_id}")
    print(f"词汇表大小 (vocab_size): {vocab_size}")
    
    return vocab_size

def print_model_parameters(model: nn.Module):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 50)
    print("模型参数统计:")
    print(f"  - 总参数量:       {total_params:,}")
    print(f"  - 可训练参数量:   {trainable_params:,}")
    print("-" * 50)

if __name__ == "__main__":
    REAL_DATA_FILE = "user2item.pkl" 
    OUTPUT_DIR = "./sasrec_test_run"
    MAX_SEQ_LEN = 20
    EMBEDDING_DIM = 512
    EVAL_EVERY_N_EPOCHS = 5
    VOCAB_SIZE = load_data_and_get_vocab_size(REAL_DATA_FILE)
    
    config = SASRecConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        embedding_dim=EMBEDDING_DIM,
        pad_token_id=0,
    )
    model = SASRec4HF(config)
    print_model_parameters(model)
    train_dataset = SASRecDataset(
        data_interaction_files=REAL_DATA_FILE,
        config=config.to_dict(),
        mode='train'
    )
    eval_dataset = SASRecDataset(
        data_interaction_files=REAL_DATA_FILE,
        config=config.to_dict(),
        mode='valid'
    )
    test_dataset = SASRecDataset(
        data_interaction_files=REAL_DATA_FILE,
        config=config.to_dict(),
        mode='test'
    )
    data_collator = SASRecDataCollator(pad_token_id=config.pad_token_id, max_seq_len=MAX_SEQ_LEN)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=200, 
        per_device_train_batch_size=2048,
        per_device_eval_batch_size=512,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="NDCG@10", 
        greater_is_better=True,       
        report_to="none",
        learning_rate=5e-4, 
        # weight_decay=0.01,
        # warmup_steps=500, 
    )
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=6), 
        EvaluateEveryNEpochsCallback(n_epochs=EVAL_EVERY_N_EPOCHS)
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks = callbacks,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    

    trainer.train()

    print("--- 开始最终评估 ---")
    test_results = trainer.evaluate(eval_dataset=test_dataset) 
    test_results_renamed = {f"test_{k}": v for k, v in test_results.items()}
    print("评估结果:", test_results_renamed)