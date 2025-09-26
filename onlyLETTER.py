import os
import torch
import json
from torch.utils.data import DataLoader
from datetime import datetime
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf

from pipelines.tokenizer_pipeline.letter_pipeline import RQVAETrainingPipeline
from genrec.datasets.model_dataset import SeqModelTrainingDataset
from genrec.tokenizers.LetterTokenizer import LetterTokenizer
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
            'tokenizer': os.path.join(base_output_dir,      os.environ["NNI_EXP_ID"], os.environ["NNI_TRIAL_JOB_ID"],'tokenizer_model'),
            'checkpoints': os.path.join(base_output_dir,    os.environ["NNI_EXP_ID"], os.environ["NNI_TRIAL_JOB_ID"], 'checkpoints'),
            'logs': os.path.join(nni_output_dir, 'logs')
        } 
    else:
        dirs = {
            'base': base_output_dir,
            'tokenizer': os.path.join(base_output_dir, 'tokenizer_model'),
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


@hydra.main(version_base=None, config_path="config", config_name="config")
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
    
    if accelerator.is_main_process:
        tokenizer_success = stage1_train_tokenizer(
            rqvae_config, output_dirs, force_retrain=cfg.force_retrain_tokenizer
        )
        if not tokenizer_success:
            logger.info("Tokenizer训练失败，终止流程")
            return
        success = success and tokenizer_success
    accelerator.wait_for_everyone() # 等待主进程完成tokenizer训练
    
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