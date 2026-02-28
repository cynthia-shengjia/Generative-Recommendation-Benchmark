#!/usr/bin/env python3
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
import nni


def get_nni_params() -> Dict[str, Any]:

    try:
        nni_params = nni.get_next_parameter()
        # logger.info(f"Using NNI parameters: {nni_params}")
        return nni_params
    except Exception as e:
        # logger.warning(f"NNI not available or not in trial: {e}")
        return {}

def update_config_with_nni(config: DictConfig, nni_params: Dict[str, Any]) -> DictConfig:

    if not nni_params:
        return config
    

    updated_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    for key, value in nni_params.items():

        if '.' in key:
            parts = key.split('.')
            current = updated_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            updated_config[key] = value
    
    return updated_config

def report_nni_metrics(metrics: Dict[str, float], is_final: bool = False, callback=None):  

    try:  

        main_metric_key = "eval_hit@5"
        current_score = metrics.get(main_metric_key, float('-inf'))
        
        if callback is not None:
            if current_score > callback.best_score:
                callback.best_score = current_score
                callback.best_metrics = metrics.copy()
                callback.best_metrics.update({"default": current_score})
        
        metrics.update({"default": current_score})
        nni.report_intermediate_result(metrics) 
        
        if is_final:
            if callback is not None and callback.best_metrics is not None:
                nni.report_final_result(callback.best_metrics)
            else:
                nni.report_final_result(metrics)
                
    except Exception as e:  
        pass 
