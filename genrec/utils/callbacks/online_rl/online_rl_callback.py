from transformers import TrainerCallback, TrainingArguments, TrainerState


class EvaluateEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs
        self.last_eval_epoch = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch ) % self.n_epochs == 0:
            control.should_evaluate = True
            self.last_eval_epoch = state.epoch
        else:
            control.should_evaluate = False
            
    def on_evaluate(self, args, state, control, metrics, **kwargs):

        control.should_save = state.epoch == self.last_eval_epoch


class GRPOLoggingCallback(TrainerCallback):
    """
    GRPO Callback
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            if any(key.startswith("eval_") for key in logs.keys()):
                self.logger.info("***** GRPO Evaluation Result *****")
                for key, value in logs.items():
                    self.logger.info(f"  {key}: {value}")
            else:
                _logs = {k: v for k, v in logs.items() if k not in ["epoch", "step"]}
                
                grpo_metrics = ["reward", "reward_std", "kl", "accuracy", "diversity", "completion_length"]
                grpo_log_items = []
                other_log_items = []
                
                for k, v in _logs.items():
                    formatted = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    if any(metric in k for metric in grpo_metrics):
                        grpo_log_items.append(formatted)
                    else:
                        other_log_items.append(formatted)
                
                log_str = f"Step {state.global_step} (Epoch {state.epoch:.2f})"
                if other_log_items:
                    log_str += " | " + " | ".join(other_log_items)
                if grpo_log_items:
                    log_str += "\n  GRPO Metric: " + " | ".join(grpo_log_items)
                
                self.logger.info(log_str)
