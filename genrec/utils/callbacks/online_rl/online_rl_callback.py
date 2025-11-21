from transformers import TrainerCallback, TrainingArguments, TrainerState


class EvaluateEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs
        self.last_eval_epoch = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # 每隔n_epochs开启评估，否则关闭
        if (state.epoch ) % self.n_epochs == 0:
            control.should_evaluate = True
            self.last_eval_epoch = state.epoch
        else:
            control.should_evaluate = False
            
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 仅在评估时保存检查点
        control.should_save = state.epoch == self.last_eval_epoch


class GRPOLoggingCallback(TrainerCallback):
    """
    GRPO 专用的日志回调函数
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            if any(key.startswith("eval_") for key in logs.keys()):
                self.logger.info("***** GRPO 验证结果 *****")
                for key, value in logs.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                # 过滤并格式化日志
                _logs = {k: v for k, v in logs.items() if k not in ["epoch", "step"]}
                
                # GRPO 特有的指标
                grpo_metrics = ["reward", "reward_std", "kl", "accuracy", "diversity", "completion_length"]
                grpo_log_items = []
                other_log_items = []
                
                for k, v in _logs.items():
                    formatted = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    if any(metric in k for metric in grpo_metrics):
                        grpo_log_items.append(formatted)
                    else:
                        other_log_items.append(formatted)
                
                log_str = f"步骤 {state.global_step} (Epoch {state.epoch:.2f})"
                if other_log_items:
                    log_str += " | " + " | ".join(other_log_items)
                if grpo_log_items:
                    log_str += "\n  GRPO指标: " + " | ".join(grpo_log_items)
                
                self.logger.info(log_str)
