
from datetime import datetime
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from genrec.utils.factory import get_pipeline_class
from genrec.utils.nni_utils import get_nni_params, update_config_with_nni
from genrec.utils.common_utils import set_seed
from genrec.utils.logging_utils import setup_logging

import hydra
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

 
def setup_output_directories(base_output_dir: str = "./output"):
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

def stage1_train_tokenizer(rqvae_config: dict, output_dirs: dict, gen_type: str,force_retrain: bool = False):
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
        PipelineClass = get_pipeline_class(gen_type)
        pipeline = PipelineClass(rqvae_config)
        pipeline.run()
        return True
    except Exception as e:
        print(f"RQ-VAE tokenizer train false: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@hydra.main(version_base=None, config_path="config", config_name="quantization")
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
        logger.info(f"type: {cfg.type}")
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
    # get type
    gen_type = cfg.type
    if accelerator.is_main_process:
        tokenizer_success = stage1_train_tokenizer(
            rqvae_config, output_dirs, gen_type=gen_type ,force_retrain=cfg.force_retrain_tokenizer
        )
        if not tokenizer_success:
            logger.info("Tokenizer train error")
            return
        success = success and tokenizer_success
    accelerator.wait_for_everyone() 
    
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