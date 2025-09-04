import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
import math

def create_custom_t5_model(
    codebook_size: int,
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
        codebook_size=codebook_size,
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
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        tie_word_embeddings=tie_word_embeddings
    )
    
    return model



def train_model(
    model,
    train_dataloader,
    dataset_name,
    num_epochs=None,
    num_steps=None,
    batch_size=256,
    learning_rate=0.01,
    device=None
):
    """
    训练模型，根据数据集名称确定训练步数
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    # 根据数据集名称确定训练步数
    if dataset_name.lower() in ["beauty", "sports and outdoors"]:
        total_steps = 200000
    elif dataset_name.lower() in ["toys and games"]:
        total_steps = 100000
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}")
    
    # 计算每个epoch的步数
    steps_per_epoch = len(train_dataloader)
    
    # 如果指定了epoch数，计算总步数
    if num_epochs is not None:
        total_steps = num_epochs * steps_per_epoch
    
    # 如果指定了步数，使用指定的步数
    if num_steps is not None:
        total_steps = num_steps
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 设置学习率调度器 - 前10k步使用固定学习率，之后使用逆平方根衰减
    def lr_lambda(current_step):
        if current_step < 10000:
            return 1.0
        else:
            return 1.0 / math.sqrt(current_step / 10000)
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练循环
    model.train()
    global_step = 0
    epoch = 0
    
    while global_step < total_steps:
        epoch += 1
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            if global_step >= total_steps:
                break
                
            # 将数据移动到设备
            encoder_input_ids = batch["encoder_input_ids"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 打印训练信息
            if global_step % 1000 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {global_step}/{total_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # 打印epoch信息
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # 保存检查点（每5个epoch）
        if epoch % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    print("Training completed!")
    return model



# 假设您已经有了数据集和codebook_size
codebook_size = 10000  # 您的codebook大小
dataset_name = "Beauty"  # 或 "Sports and Outdoors" 或 "Toys and Games"

# 创建模型
model = create_custom_t5_model(
    codebook_size=codebook_size,
    d_kv=64,           # 每个注意力头的维度
    d_ff=1024,         # 前馈网络维度
    num_layers=4,      # 编码器层数
    num_decoder_layers=4,  # 解码器层数
    num_heads=6,       # 注意力头数
    dropout_rate=0.1,  # dropout率
    tie_word_embeddings=True
)


# 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数: {total_params:,}")

# 准备数据加载器
train_dataloader = DataLoader(
    dataset,  # 您的数据集
    batch_size=256,  # 根据配置
    shuffle=True,
    num_workers=4
)

# 训练模型
trained_model = train_model(
    model=model,
    train_dataloader=train_dataloader,
    dataset_name=dataset_name,
    batch_size=256,  # 根据配置
    learning_rate=0.01  # 根据配置
)

# 保存最终模型
final_model_path = "final_model.pt"
torch.save(trained_model.state_dict(), final_model_path)
print(f"最终模型已保存到: {final_model_path}")