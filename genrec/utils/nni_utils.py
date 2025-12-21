#!/usr/bin/env python3
from typing import Dict, Any
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

def report_nni_metrics(metrics: Dict[str, float], is_final: bool = False, callback=None):  
    """向NNI报告指标"""  
    try:  
        # 确定主要评估指标
        main_metric_key = "eval_hit@5"
        current_score = metrics.get(main_metric_key, float('-inf'))
        
        # 更新最佳指标
        if callback is not None:
            if current_score > callback.best_score:
                callback.best_score = current_score
                callback.best_metrics = metrics.copy()
                callback.best_metrics.update({"default": current_score})
        
        # 报告中间结果
        metrics.update({"default": current_score})
        nni.report_intermediate_result(metrics)  # 无论是否 final 都要报告
        
        # 如果是最后一次，额外报告最终结果（使用历史最佳）
        if is_final:
            if callback is not None and callback.best_metrics is not None:
                nni.report_final_result(callback.best_metrics)
            else:
                # 如果没有跟踪到最佳指标，使用当前指标
                nni.report_final_result(metrics)
                
    except Exception as e:  
        pass  # 建议至少记录错误：logging.warning(f"NNI报告失败: {e}")
