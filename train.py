import argparse
import os
from typing import List, Dict, Any

import torch
from transformers import (
    EarlyStoppingCallback,
    T5Config,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from dataclasses import dataclass

# 从我们更新后的 utils 和 data 文件中导入
from utils import set_seed, ensure_dir, load_datasets, parse_global_args, parse_train_args, parse_dataset_args
from data import SeqRecDataset

VOCAB_SIZE_BASE = 1024
PAD_TOKEN_ID = VOCAB_SIZE_BASE
EOS_TOKEN_ID = VOCAB_SIZE_BASE + 1
TOTAL_VOCAB_SIZE = VOCAB_SIZE_BASE + 2

@dataclass
class SeqRecDataCollator:
    """
    为序列推荐任务定制的数据整理器。
    - 能处理变长的历史序列。
    - 能为历史和目标中的每一个4-token组正确应用位置偏移。
    """
    max_his_len: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for feature in features:
            # --- 处理输入（历史）序列 ---
            source_tokens = feature["source_tokens"]
            transformed_source = []
            for i, token in enumerate(source_tokens):
                offset = (i % 4) * 256
                transformed_source.append(token + offset)
            transformed_source.append(EOS_TOKEN_ID)
            if len(transformed_source) > self.max_his_len:
                    transformed_source = transformed_source[-self.max_his_len:]
            else:
                padding_length = self.max_his_len - len(transformed_source)
                transformed_source = [PAD_TOKEN_ID] * padding_length + transformed_source
            input_ids_list.append(transformed_source)

            # --- 处理目标序列 ---
            target_tokens = feature["target_tokens"] # e.g., [t1c, t2c, t3c, t4c]
            transformed_target = []
            for i, token in enumerate(target_tokens):
                offset = i * 256
                transformed_target.append(token + offset)
            transformed_target.append(EOS_TOKEN_ID)
            labels_list.append(transformed_target)

        # --- 填充 ---
        # max_input_len = max(len(ids) for ids in input_ids_list)
        # for ids in input_ids_list:
        #     ids.extend([PAD_TOKEN_ID] * (max_input_len - len(ids)))
            
        # max_label_len = max(len(lbl) for lbl in labels_list)
        # for lbl in labels_list:
        #     lbl.extend([-100] * (max_label_len - len(lbl)))
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = (input_ids_tensor != PAD_TOKEN_ID).long()
        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }

def train(args):
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1)); ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    device = torch.device("cuda", local_rank)
    if local_rank == 0: print(vars(args))
    config = AutoConfig.from_pretrained("t5-small")
    config.vocab_size = TOTAL_VOCAB_SIZE
    config.pad_token_id = PAD_TOKEN_ID
    config.eos_token_id = EOS_TOKEN_ID
    config.decoder_start_token_id = PAD_TOKEN_ID
    config.num_layers = 4
    config.num_decoder_layers = 4
    config.d_model = 384
    config.num_heads = 6
    config.d_kv = 64
    config.d_ff = 1024
    config.feed_forward_proj = "relu"
    config.dropout_rate = 0.1
    # config = T5Config(
    #     vocab_size=TOTAL_VOCAB_SIZE, pad_token_id=PAD_TOKEN_ID, eos_token_id=EOS_TOKEN_ID,
    #     decoder_start_token_id=PAD_TOKEN_ID, num_layers=4, num_decoder_layers=4,
    #     d_model=384, num_heads=6, d_kv=64, d_ff=1024, feed_forward_proj="relu", dropout_rate=0.1,
    # )
    model = T5ForConditionalGeneration(config)
    model.to(device)

    if local_rank == 0:
        config.save_pretrained(args.output_dir)

    train_data, valid_data = load_datasets(args)

    data_collator = SeqRecDataCollator(max_his_len=4*args.max_his_len)

    training_args = TrainingArguments(
        seed=args.seed, output_dir=args.output_dir, num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim, logging_steps=args.logging_step,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        ddp_find_unused_parameters=False if ddp else None,
        remove_unused_columns=False,
        weight_decay = args.weight_decay,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data,
        eval_dataset=valid_data, data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train T5 for Sequential Recommendation with 4-Token Items')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    
    train(args)

