#!/usr/bin/env python3
import os
from typing import Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf
import nni


def get_nni_params() -> Dict[str, Any]:
    """从NNI获取调优参数"""  
    try:
        nni_params = nni.get_next_parameter()
        # logger.info(f"Using NNI parameters: {nni_params}")
        return nni_params
    except Exception as e:
        # logger.warning(f"NNI not available or not in trial: {e}")
        return {}

def update_config_with_nni(config: DictConfig, nni_params: Dict[str, Any]) -> DictConfig:
    """使用NNI参数更新配置"""
    if not nni_params:
        return config
    
    # 创建配置的深拷贝
    updated_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    # 更新参数
    for key, value in nni_params.items():
        # 处理嵌套参数（如tokenizer.learning_rate）
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

def report_nni_metrics(metrics: Dict[str, float], step: int = None):
    """向NNI报告指标"""
    try:
        # 将指标报告给NNI
        for metric_name, metric_value in metrics.items():
            # NNI需要将指标作为中间结果或最终结果报告
            if step is not None:
                nni.report_intermediate_result({metric_name: metric_value})
            else:
                nni.report_final_result({metric_name: metric_value})
    except Exception as e:
        pass
        # logger.warning(f"Failed to report metrics to NNI: {e}")
