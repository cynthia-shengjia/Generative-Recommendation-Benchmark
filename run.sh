#!/bin/bash

# 设置Hydra配置路径
export HYDRA_CONFIG_PATH=conf

# 运行训练脚本
python main.py \
    dataset="Toys and Games" \
    output_dir="/home/zsj/models/test" \
    cuda="0" \
    tokenizer.learning_rate=0.4 \
    tokenizer.batch_size=1024 \
    tokenizer.epochs=2 \
    model.batch_size=256 \
    model.learning_rate=0.001 \
    model.num_epochs=2