#!/bin/bash

# 设置Hydra配置路径
export HYDRA_CONFIG_PATH=config
export CUDA_VISIBLE_DEVICES=7
tokenizer_batch_size=1024
learning_rate=0.01
model_batch_size=256
model_lr=0.001
# 运行训练脚本
python main.py \
    dataset="Toys and Games" \
    output_dir="/home/zsj/models/tokenizer-$tokenizer_batch_size-$learning_rate-model-$model_batch_size-$model_lr" \
    cuda=0 \
    tokenizer.learning_rate=$learning_rate \
    tokenizer.batch_size=$tokenizer_batch_size \
    tokenizer.epochs=20000 \
    model.batch_size=$model_batch_size \
    model.learning_rate=$model_lr \
    model.num_epochs=1
