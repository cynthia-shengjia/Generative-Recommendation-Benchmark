import os
from torch.utils.data import DataLoader
from datetime import datetime
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf


from genrec.quantization.pipelines.rqvae_pipeline import RQVAETrainingPipeline
from genrec.quantization.tokenizers.rqvae_tokenizer import RQVAETokenizer
from genrec.data.datasets.generative.tiger_dataset import TigerDataset
from genrec.data.collators.generative.tiger_collator import TigerDataCollator
from genrec.utils.nni_utils import get_nni_params, update_config_with_nni
from genrec.utils.common_utils import set_seed
from genrec.utils.logging_utils import setup_logging
from genrec.utils.evaluation_utils import evaluate_model_with_constrained_beam_search
from genrec.utils.models_setup.conditional_t5_setup import create_t5_model
from genrec.utils.trainer_setup.online_rl.grpo_setup import setup_training


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

def stage2_train_generation_model(    
    model_config,     
    rqvae_config,     
    online_rl_config,
    output_dirs,     
    accelerator,     
    logger,     
    force_retrain=False    
):
    """阶段2: 训练生成模型（使用约束beam search进行评估）"""
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        logger.info("阶段2: 训练生成模型 (Post-training)")
        logger.info("="*60)
    
    # ============ 确定模型保存路径 ============
    # 优先使用 online_rl_config 中指定的路径
    custom_save_path = online_rl_config.get('save_model_path', None)
    
    if custom_save_path:
        # 使用自定义保存路径
        model_save_dir = custom_save_path
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"{model_config['dataset_name']}_final_model.pt")
        if accelerator.is_main_process:
            logger.info(f"使用自定义保存路径: {model_save_dir}")
    else:
        # 使用默认路径
        model_save_dir = output_dirs['model']
        model_save_path = model_config['model_save_path']
        if accelerator.is_main_process:
            logger.info(f"使用默认保存路径: {model_save_dir}")
    
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
        # ============ 加载 Tokenizer ============
        if accelerator.is_main_process:
            logger.info(f"正在从 {tokenizer_object_path} 加载完整的tokenizer...")
        tokenizer = RQVAETokenizer.load(tokenizer_object_path)
        if accelerator.is_main_process:
            logger.info(f"成功加载tokenizer，包含 {len(tokenizer.item2tokens)} 个物品的token映射")
            logger.info(f"Tokenizer的完整词汇表大小: {tokenizer.vocab_size}")
              
        # ============ 加载或创建模型 ============
        pretrained_model_path = online_rl_config.get('pretrained_model', None)
          
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            # 加载预训练模型 (Hugging Face 格式)
            if accelerator.is_main_process:
                logger.info(f"🔥 从预训练模型加载: {pretrained_model_path}")
              
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
              
            if accelerator.is_main_process:
                logger.info("✅ 成功加载预训练模型")
        else:
            # 从头创建新模型
            if accelerator.is_main_process:
                if pretrained_model_path:
                    logger.warning(f"⚠️ 预训练模型路径不存在: {pretrained_model_path}")
                logger.info("从头开始创建新模型...")
              
            model = create_t5_model(
                vocab_size=tokenizer.vocab_size,
                model_config=model_config
            )

        # ============ 打印模型信息 ============
        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"模型总参数数量: {total_params:,}")
            logger.info(f"可训练参数数量: {trainable_params:,}")
            logger.info("创建数据集...")

        # ============ 创建数据集 ============
        train_dataset = TigerDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer, 
            config=model_config, 
            mode='train',
        )
          
        valid_dataset = TigerDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer, 
            config=model_config, 
            mode='valid',
        )
          
        test_dataset = TigerDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer, 
            config=model_config, 
            mode='test',
        )

        # ============ 创建 Data Collator ============
        eval_data_collator = TigerDataCollator(
            max_seq_len=train_dataset.max_token_len,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
            mode="train"
        )
          
        test_data_collator = TigerDataCollator(
            max_seq_len=train_dataset.max_token_len,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
            mode='test'
        )
     
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=model_config['test_batch_size'],
            shuffle=False,
            collate_fn=test_data_collator
        )
          
        # 使用 accelerator 准备数据加载器
        test_dataloader = accelerator.prepare(test_dataloader)
          
        # ============ 计算 Batch Size ============
        train_batch_size = model_config['batch_size']
        test_batch_size = model_config['test_batch_size']
        num_devices = accelerator.num_processes
          
        if train_batch_size % num_devices != 0 or test_batch_size % num_devices != 0:
            if accelerator.is_main_process:
                logger.error(f"错误: 训练批次大小 {train_batch_size} 或测试批次大小 {test_batch_size} 不能被设备数量 {num_devices} 整除。")
            return False
          
        per_device_train_batch_size = train_batch_size // num_devices
        per_device_eval_batch_size = test_batch_size // num_devices
          
        if accelerator.is_main_process:
            logger.info(f"Batch Size 配置 (总共 {num_devices} 个设备)")
            logger.info(f"  - 训练: 全局 {train_batch_size} -> 单设备 {per_device_train_batch_size}")
            logger.info(f"  - 评估: 全局 {test_batch_size} -> 单设备 {per_device_eval_batch_size}")
          
        # ============ 设置训练器 ============
        # 如果使用自定义保存路径，需要更新 checkpoint_dir
        if custom_save_path:
            checkpoint_dir = os.path.join(model_save_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_config['checkpoint_dir'] = checkpoint_dir
        
        trainer = setup_training(
            model,
            tokenizer,
            train_dataset,
            valid_dataset,
            model_config,
            online_rl_config,
            output_dirs,
            logger,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            train_data_collator=eval_data_collator
        )
        
        # ============ 开始训练 ============
        trainer.train()
        accelerator.wait_for_everyone()
          
        # ============ 测试评估 ============
        if accelerator.is_main_process:
            logger.info("使用约束beam search进行测试评估...")
          
        evaluate_model_with_constrained_beam_search(
            model=model,
            eval_dataloader=test_dataloader,
            accelerator=accelerator,
            tokenizer=tokenizer,
            k_list=model_config.get("k_list", [5, 10, 20]),
            num_beams=model_config.get("num_beams", 10),
            max_gen_length=model_config.get("max_gen_length", 5),
            logger=logger,
            mode="Test"
        )
          
        # ============ 保存最终模型 ============
        if "NNI_PLATFORM" not in os.environ:
            if accelerator.is_main_process:
                logger.info(f"💾 保存最终模型到: {model_save_dir}")
            
            # 保存到指定路径
            trainer.save_model(model_save_dir)
            
            if accelerator.is_main_process:
                logger.info(f"✅ 模型已保存到: {model_save_dir}")
                logger.info(f"   - Hugging Face 格式文件")
                logger.info(f"   - 检查点目录: {model_config.get('checkpoint_dir', 'N/A')}")
          
        if accelerator.is_main_process:
            logger.info("生成模型训练和评估完成!")
          
        return True
          
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"生成模型训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
        return False

@hydra.main(version_base=None, config_path="config", config_name="online_rl")
def main(cfg: DictConfig):
    """主函数"""
    seed = getattr(cfg, 'seed', 42) 
    set_seed(seed)

    if "NNI_PLATFORM" in os.environ:
        nni_params = get_nni_params()
        cfg = update_config_with_nni(cfg, nni_params)

    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device
    # 设置CUDA设备
    logger = None
    
    output_dirs = setup_output_directories(cfg.output_dir)
    if accelerator.is_main_process:
        logger = setup_logging(output_dirs['logs'])
        logger.info(f"输出目录已设置: {output_dirs['base']}")
        logger.info(f"数据集: {cfg.dataset}")
        logger.info(f"输出目录: {cfg.output_dir}")
        logger.info(f"检测到 {accelerator.num_processes} 个进程")
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
    
    if not cfg.skip_tokenizer:
        if accelerator.is_main_process:
            tokenizer_success = stage1_train_tokenizer(
                rqvae_config, output_dirs, force_retrain=cfg.force_retrain_tokenizer
            )
            if not tokenizer_success:
                logger.info("Tokenizer训练失败，终止流程")
                return
            success = success and tokenizer_success
        accelerator.wait_for_everyone() # 等待主进程完成tokenizer训练
    elif accelerator.is_main_process:
        logger.info("跳过tokenizer训练阶段")
    
    if not cfg.skip_model and success:
        # 获取模型配置
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        model_config['device'] = device
        model_config['dataset_name'] = cfg.dataset
        model_config['model_save_path'] = os.path.join(output_dirs['model'], f"{cfg.dataset}_final_model.pt")
        model_config['checkpoint_dir'] = output_dirs['checkpoints']

        online_rl_config = OmegaConf.to_container(cfg.online_rl, resolve=True)
        
        model_success = stage2_train_generation_model(
            model_config, 
            rqvae_config, 
            online_rl_config,
            output_dirs, 
            accelerator,
            force_retrain=cfg.force_retrain_model,
            logger=logger
        )
        success = success and model_success
    elif cfg.skip_model and accelerator.is_main_process:
        logger.info("跳过生成模型训练阶段")
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        if success:
            logger.info("训练流程全部完成!")
            logger.info(f"模型和检查点保存在: {output_dirs['base']}")
        else:
            logger.info("训练流程中遇到错误")
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()