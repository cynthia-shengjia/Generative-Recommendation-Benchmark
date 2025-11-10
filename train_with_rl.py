import os
import torch
import json
from torch.utils.data import DataLoader
from datetime import datetime
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf

from pipelines.tokenizer_pipeline.rqvae_pipeline import RQVAETrainingPipeline
from genrec.datasets.model_dataset import SeqModelTrainingDataset
from genrec.tokenizers.TigerTokenizer import TigerTokenizer
from genrec.datasets.data_collator import TrainSeqRecDataCollator,TestSeqRecDataCollator
from transformers import T5ForConditionalGeneration
from tools.nni_utils import get_nni_params, update_config_with_nni, report_nni_metrics
from tools.utils import set_seed, setup_logging
from tools.train_utils import evaluate_model_with_constrained_beam_search, create_t5_model
from tools.trainer_utils import setup_training
import random
import numpy as np
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_output_directories(base_output_dir: str = "./output"):
    """设置输出目录结构"""
    if "NNI_PLATFORM" in os.environ:
        nni_output_dir = os.environ["NNI_OUTPUT_DIR"]
        dirs = {
            'base': base_output_dir,
            'tokenizer': os.path.join(base_output_dir, 'tokenizer_model'),
            'model': os.path.join(base_output_dir, 'generation_model'),
            'checkpoints': os.path.join(base_output_dir, 'checkpoints'),
            'logs': os.path.join(nni_output_dir, 'logs')
        } 
    else:
        dirs = {
            'base': base_output_dir,
            'tokenizer': os.path.join(base_output_dir, 'tokenizer_model'),
            'model': os.path.join(base_output_dir, 'generation_model'),
            'checkpoints': os.path.join(base_output_dir, 'checkpoints'),
            'logs': os.path.join(base_output_dir, 'logs')
        }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def stage1_train_tokenizer(rqvae_config: dict, output_dirs: dict, force_retrain: bool = False):
    """阶段1: 训练RQ-VAE tokenizer"""
    print("\n" + "="*60)
    print("阶段1: 训练RQ-VAE Tokenizer")
    print("="*60)
    
    tokenizer_checkpoint = rqvae_config['checkpoint_path']
    item2tokens_path = rqvae_config['save_path']
    
    if not force_retrain and os.path.exists(tokenizer_checkpoint) and os.path.exists(item2tokens_path):
        print(f"发现已存在的tokenizer检查点: {tokenizer_checkpoint}")
        print("跳过tokenizer训练阶段...")
        return True
    
    required_files = [rqvae_config['data_text_files'], rqvae_config['interaction_files']]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 数据文件不存在: {file_path}")
            return False
    
    try:
        pipeline = RQVAETrainingPipeline(rqvae_config)
        pipeline.run()
        print("RQ-VAE tokenizer训练完成!")
        return True
    except Exception as e:
        print(f"RQ-VAE tokenizer训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def stage2_train_generation_model(model_config, rqvae_config, output_dirs, accelerator, logger, force_retrain=False):
    """阶段2: 训练生成模型（使用约束beam search进行评估）"""
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        logger.info("阶段2: 训练生成模型")
        logger.info("="*60)
    
    model_save_path = model_config['model_save_path']
    if not force_retrain and os.path.exists(model_save_path):
        if accelerator.is_main_process:
            logger.info(f"发现已存在的模型: {model_save_path}")
            logger.info("跳过模型训练阶段...")
        return True
    
    tokenizer_items2tokens_path = os.path.join(output_dirs['tokenizer'], 'item2tokens.json')
    if not os.path.exists(tokenizer_items2tokens_path):
        if accelerator.is_main_process:
            logger.info(f"错误: tokenizer未完成训练，找不到文件: {tokenizer_items2tokens_path}")
        return False
    tokenizer_object_path = rqvae_config['tokenizer_path']
    if not os.path.exists(tokenizer_object_path):
        if accelerator.is_main_process:
            logger.info(f"错误: 找不到完整的tokenizer对象文件: {tokenizer_object_path}")
            logger.info("请先运行阶段1进行训练。")
        return False
    
    try:
        if accelerator.is_main_process:
            logger.info(f"正在从 {tokenizer_object_path} 加载完整的tokenizer...")
        tokenizer = TigerTokenizer.load(tokenizer_object_path)
        if accelerator.is_main_process:
            logger.info(f"成功加载tokenizer，包含 {len(tokenizer.item2tokens)} 个物品的token映射")
            logger.info(f"Tokenizer的完整词汇表大小: {tokenizer.vocab_size}")
            logger.info("创建生成模型...")
        # 创建模型
        
        model = create_t5_model(
            vocab_size=tokenizer.vocab_size,
            model_config=model_config
        )

        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"模型总参数数量: {total_params:,}")
            logger.info("创建数据集...")

        # 创建数据集
        train_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer, config=model_config, mode='train'
        )
        valid_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer, config=model_config, mode='valid'
        )
        test_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer, config=model_config, mode='test'
        )

        # 创建数据整理器
        train_data_collator = TrainSeqRecDataCollator(
            max_seq_len=train_dataset.max_token_len,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
            tokens_per_item=train_dataset.tokens_per_item
        )
        
        test_data_collator = TestSeqRecDataCollator(
            max_seq_len=train_dataset.max_token_len,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
            tokens_per_item=train_dataset.tokens_per_item
        )
        # 创建验证和测试数据加载器（用于自定义评估）
 
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=model_config['test_batch_size'],
            shuffle=False,
            collate_fn=test_data_collator # 使用 HF 提供的 collator
        )
        
        # 使用accelerator准备数据加载器
        test_dataloader = accelerator.prepare(test_dataloader)
 
        trainer = setup_training(
            model, 
            tokenizer, 
            train_dataset, 
            valid_dataset, 
            model_config, 
            output_dirs, 
            train_data_collator = train_data_collator, 
            logger = logger, 
            use_generative=model_config.get('use_generative', False)
        )
 
        trainer.train()
        accelerator.wait_for_everyone() 

        
        # 使用约束beam search进行测试评估
        if accelerator.is_main_process:
            logger.info("使用约束beam search进行测试评估...")
        # test_metrics = evaluate_model_with_constrained_beam_search(
        #     model=model,
        #     eval_dataloader=test_dataloader,
        #     accelerator=accelerator,
        #     tokenizer=tokenizer,
        #     k_list=model_config.get("k_list", []),
        #     num_beams=model_config.get("num_beams", 10),
        #     max_gen_length=model_config.get("max_gen_length", 5),
        #     logger=logger,
        #     mode="Test"
        # )
        
        # 保存最终模型
        if "NNI_PLATFORM" not in os.environ:
            trainer.save_model(output_dirs['model'])
        
        if accelerator.is_main_process:
            logger.info("生成模型训练和评估完成!")
        
        return True
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"生成模型训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
        return False

# train.py 的修改部分

def stage3_grpo_training(
    model_config,
    grpo_config,
    rqvae_config,
    output_dirs,
    accelerator,
    logger,
    pretrained_model_path
):
    """
    阶段3: GRPO 强化学习训练
    
    Args:
        model_config: 模型配置
        grpo_config: GRPO 配置
        rqvae_config: Tokenizer 配置
        output_dirs: 输出目录
        accelerator: Accelerator 实例
        logger: 日志记录器
        pretrained_model_path: 预训练模型目录路径（注意是目录，不是文件）
    """
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        logger.info("阶段3: GRPO 强化学习训练")
        logger.info("="*60)
    
    # 检查预训练模型是否存在
    if pretrained_model_path is None or not os.path.exists(pretrained_model_path):
        if accelerator.is_main_process:
            logger.error(f"错误: 找不到预训练模型目录: {pretrained_model_path}")
            logger.error("请先运行阶段2进行预训练")
        return False
    
    # 检查必要的模型文件
    required_files = ['config.json', 'model.safetensors']
    for file in required_files:
        file_path = os.path.join(pretrained_model_path, file)
        if not os.path.exists(file_path):
            if accelerator.is_main_process:
                logger.error(f"错误: 找不到必要的模型文件: {file_path}")
            return False
    
    # 加载 tokenizer
    tokenizer_object_path = rqvae_config['tokenizer_path']
    if not os.path.exists(tokenizer_object_path):
        if accelerator.is_main_process:
            logger.error(f"错误: 找不到 tokenizer: {tokenizer_object_path}")
        return False
    
    try:
        if accelerator.is_main_process:
            logger.info(f"正在从 {tokenizer_object_path} 加载 tokenizer...")
        
        from genrec.tokenizers.TigerTokenizer import TigerTokenizer
        tokenizer = TigerTokenizer.load(tokenizer_object_path)
        
        if accelerator.is_main_process:
            logger.info(f"成功加载 tokenizer，包含 {len(tokenizer.item2tokens)} 个物品")
            logger.info("从预训练检查点加载模型...")
        
        # 使用 from_pretrained 加载模型
        from transformers import T5ForConditionalGeneration
        
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            local_files_only=True
        )
        
        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"预训练模型加载完成，参数量: {total_params:,}")
            logger.info(f"模型配置: {model.config}")
            logger.info("创建数据集...")
        
        # 创建数据集 - 使用相同的训练集进行 GRPO 训练
        from genrec.datasets.model_dataset import SeqModelTrainingDataset
        
        train_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='train'  # 使用训练集
        )
        
        eval_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='valid'  # 使用验证集
        )
        
        test_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='test'
        )
        
        if accelerator.is_main_process:
            logger.info(f"训练集样本数: {len(train_dataset)}")
            logger.info(f"验证集样本数: {len(eval_dataset)}")
            logger.info(f"测试集样本数: {len(test_dataset)}")
            logger.info("设置 GRPO 训练...")
        
        # 设置 GRPO 训练
        from tools.grpo_trainer_utils import setup_grpo_training
        
        trainer = setup_grpo_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            grpo_config=grpo_config,
            output_dirs=output_dirs,
            logger=logger,
            pretrained_model_path=None  # 已经加载了模型
        )
        
        if accelerator.is_main_process:
            logger.info("开始 GRPO 训练...")
        
        # 训练
        trainer.train()
        
        accelerator.wait_for_everyone()
        
        # 评估
        if accelerator.is_main_process:
            logger.info("GRPO 训练完成，开始最终评估...")
        
        # 使用约束 beam search 进行测试评估
        from torch.utils.data import DataLoader
        from genrec.datasets.data_collator import TestSeqRecDataCollator
        from tools.train_utils import evaluate_model_with_constrained_beam_search
        
        test_data_collator = TestSeqRecDataCollator(
            max_seq_len=train_dataset.max_token_len,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
            tokens_per_item=train_dataset.tokens_per_item
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=grpo_config.get('test_batch_size', 8),
            shuffle=False,
            collate_fn=test_data_collator
        )
        
        test_dataloader = accelerator.prepare(test_dataloader)
        
        test_metrics = evaluate_model_with_constrained_beam_search(
            model=model,
            eval_dataloader=test_dataloader,
            accelerator=accelerator,
            tokenizer=tokenizer,
            k_list=grpo_config.get("k_list", [1, 5, 10]),
            num_beams=grpo_config.get("num_beams", 10),
            max_gen_length=grpo_config.get("max_gen_length", 5),
            logger=logger,
            mode="GRPO Test"
        )
        
        # 保存最终模型（使用 Hugging Face 格式）
        if accelerator.is_main_process:
            grpo_model_save_dir = os.path.join(
                output_dirs['grpo_model'],
                "final_model"
            )
            os.makedirs(grpo_model_save_dir, exist_ok=True)
            
            # 保存模型和配置
            model.save_pretrained(grpo_model_save_dir)
            
            logger.info(f"GRPO 模型已保存到: {grpo_model_save_dir}")
            logger.info("保存的文件包括:")
            for file in os.listdir(grpo_model_save_dir):
                logger.info(f"  - {file}")
            logger.info("GRPO 训练和评估完成!")
        
        return True
        
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"GRPO 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
        return False


# train.py 的修改部分

def stage3_grpo_training(
    model_config,
    grpo_config,
    rqvae_config,
    output_dirs,
    accelerator,
    logger,
    pretrained_model_path
):
    """
    阶段3: GRPO 强化学习训练
    
    Args:
        model_config: 模型配置
        grpo_config: GRPO 配置
        rqvae_config: Tokenizer 配置
        output_dirs: 输出目录
        accelerator: Accelerator 实例
        logger: 日志记录器
        pretrained_model_path: 预训练模型目录路径（注意是目录，不是文件）
    """
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        logger.info("阶段3: GRPO 强化学习训练")
        logger.info("="*60)
    
    # 检查预训练模型是否存在
    if pretrained_model_path is None or not os.path.exists(pretrained_model_path):
        if accelerator.is_main_process:
            logger.error(f"错误: 找不到预训练模型目录: {pretrained_model_path}")
            logger.error("请先运行阶段2进行预训练")
        return False
    
    # 检查必要的模型文件
    required_files = ['config.json', 'model.safetensors']
    for file in required_files:
        file_path = os.path.join(pretrained_model_path, file)
        if not os.path.exists(file_path):
            if accelerator.is_main_process:
                logger.error(f"错误: 找不到必要的模型文件: {file_path}")
            return False
    
    # 加载 tokenizer
    tokenizer_object_path = rqvae_config['tokenizer_path']
    if not os.path.exists(tokenizer_object_path):
        if accelerator.is_main_process:
            logger.error(f"错误: 找不到 tokenizer: {tokenizer_object_path}")
        return False
    
    try:
        if accelerator.is_main_process:
            logger.info(f"正在从 {tokenizer_object_path} 加载 tokenizer...")
        
        from genrec.tokenizers.TigerTokenizer import TigerTokenizer
        tokenizer = TigerTokenizer.load(tokenizer_object_path)
        
        if accelerator.is_main_process:
            logger.info(f"成功加载 tokenizer，包含 {len(tokenizer.item2tokens)} 个物品")
            logger.info("从预训练检查点加载模型...")
        
        # 使用 from_pretrained 加载模型
        from transformers import T5ForConditionalGeneration
        
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_path,
            local_files_only=True
        )
        
        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"预训练模型加载完成，参数量: {total_params:,}")
            logger.info(f"模型配置: {model.config}")
            logger.info("创建数据集...")
        
        # 创建数据集 - 使用相同的训练集进行 GRPO 训练
        from genrec.datasets.model_dataset import SeqModelTrainingDataset
        
        train_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='train'  # 使用训练集
        )
        
        eval_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='valid'  # 使用验证集
        )
        
        test_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='test'
        )
        
        if accelerator.is_main_process:
            logger.info(f"训练集样本数: {len(train_dataset)}")
            logger.info(f"验证集样本数: {len(eval_dataset)}")
            logger.info(f"测试集样本数: {len(test_dataset)}")
            logger.info("设置 GRPO 训练...")
        
        # 设置 GRPO 训练
        from tools.grpo_trainer_utils import setup_grpo_training
        
        trainer = setup_grpo_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            grpo_config=grpo_config,
            output_dirs=output_dirs,
            logger=logger,
            pretrained_model_path=None  # 已经加载了模型
        )
        
        if accelerator.is_main_process:
            logger.info("开始 GRPO 训练...")
        
        # 训练
        trainer.train()
        
        accelerator.wait_for_everyone()
        
        # 评估
        if accelerator.is_main_process:
            logger.info("GRPO 训练完成，开始最终评估...")
        
        # 使用约束 beam search 进行测试评估
        from torch.utils.data import DataLoader
        from genrec.datasets.data_collator import TestSeqRecDataCollator
        from tools.train_utils import evaluate_model_with_constrained_beam_search
        
        test_data_collator = TestSeqRecDataCollator(
            max_seq_len=train_dataset.max_token_len,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
            tokens_per_item=train_dataset.tokens_per_item
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=grpo_config.get('test_batch_size', 8),
            shuffle=False,
            collate_fn=test_data_collator
        )
        
        test_dataloader = accelerator.prepare(test_dataloader)
        
        test_metrics = evaluate_model_with_constrained_beam_search(
            model=model,
            eval_dataloader=test_dataloader,
            accelerator=accelerator,
            tokenizer=tokenizer,
            k_list=grpo_config.get("k_list", [1, 5, 10]),
            num_beams=grpo_config.get("num_beams", 10),
            max_gen_length=grpo_config.get("max_gen_length", 5),
            logger=logger,
            mode="GRPO Test"
        )
        
        # 保存最终模型（使用 Hugging Face 格式）
        if accelerator.is_main_process:
            grpo_model_save_dir = os.path.join(
                output_dirs['grpo_model'],
                "final_model"
            )
            os.makedirs(grpo_model_save_dir, exist_ok=True)
            
            # 保存模型和配置
            model.save_pretrained(grpo_model_save_dir)
            
            logger.info(f"GRPO 模型已保存到: {grpo_model_save_dir}")
            logger.info("保存的文件包括:")
            for file in os.listdir(grpo_model_save_dir):
                logger.info(f"  - {file}")
            logger.info("GRPO 训练和评估完成!")
        
        return True
        
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"GRPO 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
        return False


@hydra.main(version_base=None, config_path="config", config_name="rl_config")
def main(cfg: DictConfig):
    """主函数"""
    seed = getattr(cfg, 'seed', 42)
    set_seed(seed)
    
    if "NNI_PLATFORM" in os.environ:
        nni_params = get_nni_params()
        cfg = update_config_with_nni(cfg, nni_params)
    
    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device
    logger = None
    
    output_dirs = setup_output_directories(cfg.output_dir)
    
    # 添加 GRPO 输出目录
    output_dirs['grpo_model'] = os.path.join(output_dirs['base'], 'grpo_model')
    os.makedirs(output_dirs['grpo_model'], exist_ok=True)
    
    if accelerator.is_main_process:
        logger = setup_logging(output_dirs['logs'])
        logger.info(f"输出目录已设置: {output_dirs['base']}")
        logger.info(f"数据集: {cfg.dataset}")
        logger.info(f"输出目录: {cfg.output_dir}")
        logger.info(f"当前进程运行设备: {device}")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"日志文件保存在: {output_dirs['logs']}")
    
    success = True
    
    # 获取RQ-VAE配置
    rqvae_config = OmegaConf.to_container(cfg.tokenizer, resolve=True)
    rqvae_config['device'] = device
    rqvae_config['tokenizer_path'] = os.path.join(output_dirs['tokenizer'], 'tokenizer.pkl')
    rqvae_config['save_path'] = os.path.join(output_dirs['tokenizer'], 'item2tokens.json')
    rqvae_config['checkpoint_path'] = os.path.join(output_dirs['tokenizer'], 'tokenizer_checkpoint.pth')
    
    # Stage 1: Tokenizer 训练
    if not cfg.skip_tokenizer:
        if accelerator.is_main_process:
            tokenizer_success = stage1_train_tokenizer(
                rqvae_config, output_dirs, force_retrain=cfg.force_retrain_tokenizer
            )
            if not tokenizer_success:
                logger.info("Tokenizer训练失败，终止流程")
                return
            success = success and tokenizer_success
        accelerator.wait_for_everyone()
    elif accelerator.is_main_process:
        logger.info("跳过tokenizer训练阶段")
    
    # Stage 2: 预训练模型
    pretrained_model_dir = None  # 改为目录路径
    if not cfg.skip_model and success:
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        model_config['device'] = device
        model_config['dataset_name'] = cfg.dataset
        # 注意：这里不再需要 model_save_path，Trainer 会自动保存到 output_dir
        model_config['checkpoint_dir'] = output_dirs['checkpoints']
        
        model_success = stage2_train_generation_model(
            model_config, rqvae_config, output_dirs, accelerator,
            force_retrain=cfg.force_retrain_model, logger=logger
        )
        success = success and model_success
        
        if model_success:
            # Trainer 保存的模型在 output_dirs['model'] 目录下
            pretrained_model_dir = output_dirs['model']
            
            if accelerator.is_main_process:
                logger.info(f"预训练模型保存在: {pretrained_model_dir}")
                
    elif cfg.skip_model and accelerator.is_main_process:
        logger.info("跳过生成模型训练阶段")
        # 如果跳过训练，使用已有的模型目录
        pretrained_model_dir = output_dirs['model']
        
        # 检查模型是否存在
        if not os.path.exists(os.path.join(pretrained_model_dir, 'config.json')):
            logger.error(f"错误: 找不到预训练模型: {pretrained_model_dir}")
            logger.error("请先运行 Stage 2 或设置正确的模型路径")
            pretrained_model_dir = None
    
    # Stage 3: GRPO 训练
    if not cfg.get('skip_grpo', False) and success and pretrained_model_dir:
        if accelerator.is_main_process:
            logger.info("\n准备进行 GRPO 训练...")
            # 优先使用配置文件中指定的路径
        if hasattr(cfg.post_training, 'pretrained_model_path') and cfg.post_training.pretrained_model_path:
            pretrained_model_dir = cfg.post_training.pretrained_model_path
            if accelerator.is_main_process:
                logger.info(f"使用配置文件指定的预训练模型: {pretrained_model_dir}")
        if pretrained_model_dir:
            # 获取 GRPO 配置
            grpo_config = OmegaConf.to_container(cfg.post_training, resolve=True)
            model_config = OmegaConf.to_container(cfg.model, resolve=True)
            # 合并模型配置（GRPO 需要一些模型配置）
            grpo_config.update({
                'data_interaction_files': model_config['data_interaction_files'],
                'data_text_files': model_config['data_text_files'],
                'max_seq_len': model_config.get('max_seq_len', 50),
                'dataset_name': cfg.dataset,
            })
            
            grpo_success = stage3_grpo_training(
                model_config=model_config,
                grpo_config=grpo_config,
                rqvae_config=rqvae_config,
                output_dirs=output_dirs,
                accelerator=accelerator,
                logger=logger,
                pretrained_model_path=pretrained_model_dir  # 传递目录路径
            )
            success = success and grpo_success
    elif cfg.get('skip_grpo', False) and accelerator.is_main_process:
        logger.info("跳过 GRPO 训练阶段")
    elif not pretrained_model_dir and accelerator.is_main_process:
        logger.warning("未找到预训练模型，跳过 GRPO 训练")
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        if success:
            logger.info("训练流程全部完成!")
            logger.info(f"模型和检查点保存在: {output_dirs['base']}")
            if pretrained_model_dir:
                logger.info(f"  - 预训练模型: {pretrained_model_dir}")
            if not cfg.get('skip_grpo', False):
                logger.info(f"  - GRPO 模型: {output_dirs['grpo_model']}/final_model")
        else:
            logger.info("训练流程中遇到错误")
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
    accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()