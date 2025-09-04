import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from genrec.models.CustomT5.T5Model import CustomT5ForConditionalGeneration
from genrec.models.CustomT5.T5Config import CustomT5Config
import logging
import re
def create_custom_t5_model(
    vocab_size: int,
    d_kv: int = 64,         # 每个注意力头的维度
    d_ff: int = 1024,       # 前馈网络维度
    num_layers: int = 4,    # 编码器层数
    num_decoder_layers: int = 4,  # 解码器层数
    num_heads: int = 6,     # 注意力头数
    dropout_rate: float = 0.1,  # dropout率
    tie_word_embeddings: bool = True
) -> CustomT5ForConditionalGeneration:
    """
    创建自定义T5模型，根据提供的配置参数
    d_model 会自动计算为 d_kv * num_heads
    """
    config = CustomT5Config(
        vocab_size=vocab_size,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        tie_word_embeddings=tie_word_embeddings
    )
    
    # 计算 latent_dim = d_model = d_kv * num_heads
    latent_dim = d_kv * num_heads
    
    model = CustomT5ForConditionalGeneration(
        config=config,
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        tie_word_embeddings=tie_word_embeddings
    )
    
    return model

def train_model(
    model: CustomT5ForConditionalGeneration,
    train_dataloader: torch.utils.data.DataLoader,
    learning_rate: float,
    num_epochs: int,
    checkpoint_dir: str,
    dataset_name: str,
    accelerator: Accelerator,
    log_interval: int = 100,
    logger: logging.Logger = None,
    num_steps: int = None,
) -> CustomT5ForConditionalGeneration:
    """
    使用Accelerate进行模型训练，并按Epoch保存和加载检查点。
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if num_steps is None:
        total_steps = len(train_dataloader) * num_epochs
    else:
        total_steps = num_steps
        # 如果指定了总步数，num_epochs 仍然用于计算 scheduler，但循环将提前退出
        num_epochs = (total_steps // len(train_dataloader)) + 1 if len(train_dataloader) > 0 else 1

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    full_checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
    os.makedirs(full_checkpoint_dir, exist_ok=True)
    
    start_epoch = 0 
    checkpoint_folders = [d for d in os.listdir(full_checkpoint_dir) if d.startswith('epoch_')]
    if checkpoint_folders:
        latest_epoch = max([int(re.search(r'epoch_(\d+)', d).group(1)) for d in checkpoint_folders])
        latest_checkpoint_path = os.path.join(full_checkpoint_dir, f"epoch_{latest_epoch}")
        if accelerator.is_main_process and latest_checkpoint_path:
            logger.info(f"发现最新检查点，从 {latest_checkpoint_path} 恢复训练...")
            accelerator.load_state(latest_checkpoint_path)
            start_epoch = latest_epoch 
            logger.info(f"成功恢复！将从 epoch {start_epoch + 1} 开始训练。")

    
    global_step = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not accelerator.is_main_process)
        
        for batch in progress_bar:
            if num_steps is not None and global_step >= num_steps:
                break

            outputs = model(
                encoder_input_ids               = batch['encoder_input_ids'],
                encoder_attention_mask          = batch['encoder_attention_mask'],
                decoder_input_ids               = batch['decoder_input_ids'],
                decoder_attention_mask          = batch['decoder_attention_mask'],
                labels                          = batch['labels']
            )
            
            loss = outputs['loss']
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            if accelerator.is_main_process:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'step': global_step})

        accelerator.wait_for_everyone()
        save_path = os.path.join(full_checkpoint_dir, f"epoch_{epoch + 1}")
        accelerator.save_state(save_path)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1} 完成。检查点已保存至: {save_path}")

        if num_steps is not None and global_step >= num_steps and accelerator.is_main_process:
            logger.info(f"已达到目标步数 {num_steps}，训练提前结束。")
            break

    if accelerator.is_main_process:
        logger.info("\n训练完成。")
    return accelerator.unwrap_model(model)