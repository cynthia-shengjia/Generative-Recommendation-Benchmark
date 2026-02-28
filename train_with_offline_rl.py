# train_with_offline_rl.py

import os
from torch.utils.data import DataLoader
from datetime import datetime
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf

from genrec.quantization.pipelines.rqvae_pipeline import RQVAETrainingPipeline
from genrec.quantization.tokenizers.rqvae_tokenizer import RQVAETokenizer
from genrec.data.datasets.offline_rl.sdpo_dataset import SDPODataset
from genrec.data.collators.offline_rl.sdpo_collator import SDPODataCollator
from genrec.utils.nni_utils import get_nni_params, update_config_with_nni
from genrec.utils.common_utils import set_seed
from genrec.utils.logging_utils import setup_logging
from genrec.utils.evaluation_utils import evaluate_model_with_constrained_beam_search
from genrec.utils.models_setup.tiger_setup import create_tiger_model
from genrec.utils.trainer_setup.offline_rl_setup import setup_training  

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_output_directories(base_output_dir: str = "./output"):
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
    print("\n" + "="*60)
    print("RQ-VAE Tokenizer")
    print("="*60)
    
    tokenizer_checkpoint = rqvae_config['checkpoint_path']
    item2tokens_path = rqvae_config['save_path']
    
    if not force_retrain and os.path.exists(tokenizer_checkpoint) and os.path.exists(item2tokens_path):
        print(f"exist tokenizer checkpoint: {tokenizer_checkpoint}")
        print("skip tokenizer training...")
        return True
    
    required_files = [rqvae_config['data_text_files'], rqvae_config['interaction_files']]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"not exist: {file_path}")
            return False
    
    try:
        pipeline = RQVAETrainingPipeline(rqvae_config)
        pipeline.run()
        return True
    except Exception as e:
        print(f"RQ-VAE tokenizer train false: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def stage2_train_generation_model(
    model_config,
    rqvae_config,
    offline_rl_config: DictConfig, 
    output_dirs,
    accelerator,
    logger,
    force_retrain=False
):
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        logger.info("Post-training with Offline RL")
        logger.info("="*60)
    
    custom_save_path = offline_rl_config.get('save_model_path', None)
    
    if custom_save_path:
        model_save_dir = custom_save_path
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"{model_config['dataset_name']}_final_model.pt")
        if accelerator.is_main_process:
            logger.info(f"custom save path: {model_save_dir}")
    else:

        model_save_dir = output_dirs['model']
        model_save_path = model_config['model_save_path']
        if accelerator.is_main_process:
            logger.info(f"default save path: {model_save_dir}")
    
    if not force_retrain and os.path.exists(model_save_path):
        if accelerator.is_main_process:
            logger.info(f"exist model: {model_save_path}")
            logger.info("skip generation model...")
        return True
    
    tokenizer_items2tokens_path = os.path.join(output_dirs['tokenizer'], 'item2tokens.json')
    if not os.path.exists(tokenizer_items2tokens_path):
        if accelerator.is_main_process:
            logger.info(f"Error: Not Found: {tokenizer_items2tokens_path}")
        return False
    
    tokenizer_object_path = rqvae_config['tokenizer_path']
    if not os.path.exists(tokenizer_object_path):
        if accelerator.is_main_process:
            logger.info(f"Error: Not Found: {tokenizer_object_path}")
        return False
    
    try:
        if accelerator.is_main_process:
            logger.info(f"loading from {tokenizer_object_path} 加载完整的tokenizer...")
        tokenizer = RQVAETokenizer.load(tokenizer_object_path)
        if accelerator.is_main_process:
            logger.info(f"total {len(tokenizer.item2tokens)} 个物品的token映射")
            logger.info(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
        
        pretrained_model_path = offline_rl_config.get('pretrained_model', None)
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            if accelerator.is_main_process:
                logger.info(f"load from: {pretrained_model_path}")
            
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
            
            if accelerator.is_main_process:
                logger.info("✅ load finish")
        else:
            if accelerator.is_main_process:
                if pretrained_model_path:
                    logger.warning(f"not Found: {pretrained_model_path}")
                logger.info("create new model...")
            
            model = create_tiger_model(
                vocab_size=tokenizer.vocab_size,
                model_config=model_config
            )
        
        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"model parameters: {total_params:,}")
            logger.info(f"trainable parameters: {trainable_params:,}")
        
        train_dataset = SDPODataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='train',
            neg_num=offline_rl_config.get("neg_num", 4)
        )
        
        valid_dataset = SDPODataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='valid',
            neg_num=offline_rl_config.get("neg_num", 4)
        )
        
        test_dataset = SDPODataset(
            data_interaction_files=model_config['data_interaction_files'],
            data_text_files=model_config['data_text_files'],
            tokenizer=tokenizer,
            config=model_config,
            mode='test',
            neg_num=offline_rl_config.get("neg_num", 4)
        )
        
        train_data_collator = SDPODataCollator(
            max_seq_len=train_dataset.max_token_len,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
            mode="train"
        )
        
        eval_data_collator = SDPODataCollator(
            max_seq_len=train_dataset.max_token_len,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
            mode="test"
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=model_config['test_batch_size'],
            shuffle=False,
            collate_fn=eval_data_collator
        )
        
        test_dataloader = accelerator.prepare(test_dataloader)
        
        train_batch_size = model_config['batch_size']
        test_batch_size = model_config['test_batch_size']
        num_devices = accelerator.num_processes
        
        if train_batch_size % num_devices != 0 or test_batch_size % num_devices != 0:
            if accelerator.is_main_process:
                logger.error(f"Error:  {train_batch_size} or {test_batch_size} can not divide by{num_devices}")
            return False
        
        per_device_train_batch_size = train_batch_size // num_devices
        per_device_eval_batch_size = test_batch_size // num_devices
        
        if accelerator.is_main_process:
            logger.info(f"Batch Size setting (total {num_devices} devices)")
            logger.info(f"  - training global {train_batch_size} -> one device {per_device_train_batch_size}")
            logger.info(f"  - evaluation: global {test_batch_size} -> one device {per_device_eval_batch_size}")

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
            offline_rl_config, 
            output_dirs,
            logger,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            train_data_collator=train_data_collator,
            eval_data_collator=eval_data_collator,
        )
        
        trainer.train()
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            logger.info("predict test set...")
        test_results = trainer.predict(test_dataset)
        if accelerator.is_main_process:
            metrics = test_results.metrics
            
            k_values = sorted(list(set(
                int(key.split('@')[-1]) for key in metrics.keys() if '@' in key
            )))

            logger.info("="*30 + " test results " + "="*30)
            
            for k in k_values:
                hit_val = metrics.get(f"test_hit@{k}", 0.0)
                ndcg_val = metrics.get(f"test_ndcg@{k}", 0.0)
                
                logger.info(f"Hit@{k}: {hit_val:.4f}, NDCG@{k}: {ndcg_val:.4f}")
                
            logger.info("="*75)
        if "NNI_PLATFORM" not in os.environ:
            if accelerator.is_main_process:
                logger.info(f"💾 save: {model_save_dir}")
            
            trainer.save_model(model_save_dir)
            
            if accelerator.is_main_process:
                logger.info(f"✅ save: {model_save_dir}")
                logger.info(f"checkpoint: {model_config.get('checkpoint_dir', 'N/A')}")
        
        if accelerator.is_main_process:
            logger.info("Evaluation Finish!")
        
        return True
    
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        return False

@hydra.main(version_base=None, config_path="config", config_name="offline_rl")
def main(cfg: DictConfig):
    seed = getattr(cfg, 'seed', 42)
    set_seed(seed)
    
    if "NNI_PLATFORM" in os.environ:
        nni_params = get_nni_params()
        cfg = update_config_with_nni(cfg, nni_params)
    
    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device
    logger = None
    
    output_dirs = setup_output_directories(cfg.output_dir)
    if accelerator.is_main_process:
        logger = setup_logging(output_dirs['logs'])
        logger.info(f"output_dirs: {output_dirs['base']}")
        logger.info(f"dataset: {cfg.dataset}")
        logger.info(f"output_dir: {cfg.output_dir}")
        logger.info(f"{accelerator.num_processes} processes")
        logger.info(f"start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{output_dirs['logs']}")
    success = True
    
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
                logger.info("Tokenizer train error")
                return
            success = success and tokenizer_success
        accelerator.wait_for_everyone()
    elif accelerator.is_main_process:
        logger.info("skip tokenizer training")
    
    if not cfg.skip_model and success:
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        model_config['device'] = device
        model_config['dataset_name'] = cfg.dataset
        model_config['model_save_path'] = os.path.join(output_dirs['model'], f"{cfg.dataset}_final_model.pt")
        model_config['checkpoint_dir'] = output_dirs['checkpoints']
        
        model_success = stage2_train_generation_model(
            model_config,
            rqvae_config,
            cfg.offline_rl,
            output_dirs,
            accelerator,
            force_retrain=cfg.force_retrain_model,
            logger=logger
        )
        success = success and model_success
    elif cfg.skip_model and accelerator.is_main_process:
        logger.info("skip train")
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        if success:
            logger.info("Finish Train!")
            logger.info(f"checkpoint : {output_dirs['base']}")
        else:
            logger.info("Error")
        logger.info(f"Finish Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
    accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()