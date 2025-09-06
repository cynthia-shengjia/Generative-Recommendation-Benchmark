import os
import torch
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from accelerate import Accelerator
import logging 
from rqvaePipeline import RQVAETrainingPipeline
from model_train import create_custom_t5_model, train_model, test_model
from genrec.datasets.model_dataset import SeqModelTrainingDataset
from genrec.tokenizers.TigerTokenizer import TigerTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def setup_logging(log_dir: str):
    log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
def setup_output_directories(base_output_dir: str = "./output"):
    """设置输出目录结构"""
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

def get_rqvae_config(output_dirs: dict, device: torch.device, config_overrides: dict = None):
    """获取RQ-VAE tokenizer的配置"""
    config = {
        'data_text_files': './data/Beauty/item2title.pkl',
        'text_encoder_model': '/home/zsj/models/sentence-t5-base/sentence-t5-base',
        'tokenizer_path':os.path.join(output_dirs['tokenizer'], 'tokenizer.pkl'),
        'interaction_files': './data/Beauty/user2item.pkl',
        'save_path': os.path.join(output_dirs['tokenizer'], 'item2tokens.json'),
        'checkpoint_path': os.path.join(output_dirs['tokenizer'], 'tokenizer_checkpoint.pth'),
        'sent_emb_dim': 768, 'n_codebooks': 3, 'codebook_size': 256,
        'rq_e_dim': 32, 'rq_layers': [512, 256, 128], 'dropout_prob': 0.1,
        'loss_type': 'mse', 'quant_loss_weight': 1.0, 'commitment_beta': 0.25,
        'rq_kmeans_init': False, 'kmeans_iters': 10, 'learning_rate': 1e-3,
        'epochs': 2, 'batch_size': 128, 'device': device, 'log_interval': 10,
        'embedding_strategy': "mean_pooling",
    }
    if config_overrides: config.update(config_overrides)
    return config

def get_model_config(output_dirs: dict, dataset_name: str, device: torch.device, config_overrides: dict = None):
    """获取生成模型的配置"""
    config = {
        'dataset_name': dataset_name,
        'data_interaction_files': './data/Beauty/user2item.pkl',
        'data_text_files': './data/Beauty/item2title.pkl',
        'max_seq_len': 20, 'padding_side': 'left', 'ignored_label': -100,
        'vocab_size': 256 * 4, 'd_kv': 64, 'd_ff': 1024, 'num_layers': 4,
        'num_decoder_layers': 4, 'num_heads': 6, 'dropout_rate': 0.1,
        'tie_word_embeddings': True, 'batch_size': 128, 'test_batch_size': 1024, 'learning_rate': 0.001,
        'num_epochs': 2, 'num_steps': None,
        'model_save_path': os.path.join(output_dirs['model'], f'{dataset_name}_final_model.pt'),
        'checkpoint_dir': output_dirs['checkpoints'], 'device': device,
    }
    if config_overrides: config.update(config_overrides)
    return config

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

def stage2_train_generation_model(model_config: dict, rqvae_config: dict, output_dirs: dict, accelerator: Accelerator,logger, force_retrain: bool = False):
    """阶段2: 训练生成模型"""
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
        
        tokenizer.rq_vae.to(accelerator.device)
        if accelerator.is_main_process:
            logger.info(f"成功加载tokenizer，包含 {len(tokenizer.item2tokens)} 个物品的token映射")
        
        if accelerator.is_main_process:
            logger.info(f"Tokenizer的完整词汇表大小: {tokenizer.vocab_size}")
            logger.info("创建生成模型...")

        model = create_custom_t5_model(
            vocab_size=tokenizer.vocab_size, d_kv=model_config['d_kv'], d_ff=model_config['d_ff'],
            num_layers=model_config['num_layers'], num_decoder_layers=model_config['num_decoder_layers'],
            num_heads=model_config['num_heads'], dropout_rate=model_config['dropout_rate'],
            tie_word_embeddings=model_config['tie_word_embeddings']
        )
        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"模型总参数数量: {total_params:,}")
            logger.info("创建数据集...")

        train_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer, config=model_config, mode='train'
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=4
        )
        if accelerator.is_main_process:
            logger.info(f"数据集大小: {len(train_dataset)} 个样本")
            logger.info(f"批次数量: {len(train_dataloader)} 个批次")
            logger.info("开始训练生成模型...")

        trained_model = train_model(
            model=model, train_dataloader=train_dataloader,
            learning_rate=model_config['learning_rate'], num_epochs=model_config['num_epochs'],
            num_steps=model_config.get('num_steps'), checkpoint_dir=model_config['checkpoint_dir'],
            dataset_name=model_config['dataset_name'], accelerator=accelerator,logger=logger,
        )
        

        test_dataset = SeqModelTrainingDataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer, config=model_config, mode='test'
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=model_config['test_batch_size'], shuffle=False, num_workers=4
        )

        metrics = test_model(
            model = trained_model,
            test_dataloader = test_dataloader,
            accelerator = accelerator,
            tokenizer = tokenizer,
            k_list = [1,5,10],
            num_beams = 10,
            max_gen_length = tokenizer.digits + 1,
            logger=logger
        )

        if accelerator.is_main_process:
            logger.info(f"保存最终模型到: {model_save_path}")
            torch.save(trained_model.state_dict(), model_save_path)

            config_save_path = model_save_path.replace('.pt', '_config.json')
            config_to_save = model_config.copy()
            config_to_save['vocab_size'] = tokenizer.vocab_size
            if 'device' in config_to_save and not isinstance(config_to_save['device'], str):
                config_to_save['device'] = str(config_to_save['device'])
            with open(config_save_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            logger.info("生成模型训练完成!")
        
        return True
    except Exception as e:
        if accelerator.is_main_process:
            logger.info(f"生成模型训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
        return False

def main():
    """主函数"""
    accelerator = Accelerator()
    device = accelerator.device

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='Beauty', choices=['Beauty', 'Sports and Outdoors', 'Toys and Games'], help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='/home/zsj/models/TIGGER', help='输出目录')
    parser.add_argument('--force_retrain_tokenizer', action='store_true', help='强制重新训练tokenizer')
    parser.add_argument('--force_retrain_model', action='store_true', help='强制重新训练生成模型')
    parser.add_argument('--skip_tokenizer', action='store_true', help='跳过tokenizer训练')
    parser.add_argument('--skip_model', action='store_true', help='跳过生成模型训练')
    
    args = parser.parse_args()
    
    logger = None
    
    output_dirs = setup_output_directories(args.output_dir)
    if accelerator.is_main_process:
        logger = setup_logging(output_dirs['logs'])
        logger.info(f"输出目录已设置: {output_dirs['base']}")
        logger.info(f"数据集: {args.dataset}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"当前进程运行设备: {device}")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"日志文件保存在: {output_dirs['logs']}")
    
    success = True
    
    rqvae_config = get_rqvae_config(output_dirs, device)
    
    if not args.skip_tokenizer:
        if accelerator.is_main_process:
            tokenizer_success = stage1_train_tokenizer(
                rqvae_config, output_dirs, force_retrain=args.force_retrain_tokenizer
            )
            if not tokenizer_success:
                logger.info("Tokenizer训练失败，终止流程")
                return
            success = success and tokenizer_success
        accelerator.wait_for_everyone() # 等待主进程完成tokenizer训练
    elif accelerator.is_main_process:
        logger.info("跳过tokenizer训练阶段")
    
    if not args.skip_model and success:
        model_config = get_model_config(output_dirs, args.dataset, device)
        model_success = stage2_train_generation_model(
            model_config, rqvae_config, output_dirs, accelerator,
            force_retrain=args.force_retrain_model,logger=logger
        )
        success = success and model_success
    elif args.skip_model and accelerator.is_main_process:
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