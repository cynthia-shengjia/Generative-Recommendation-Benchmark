from transformers import TrainerCallback, TrainingArguments, TrainerState
from genrec.utils.nni_utils import report_nni_metrics
import os

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


class GenerativeLoggingCallback(TrainerCallback):

    def __init__(self, logger):  
        super().__init__()  
        self.logger = logger  
        self.best_metrics = None
        self.best_score = float('-inf')
  
    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):  
        if state.is_world_process_zero and logs:  
            if any(key.startswith("eval_") for key in logs.keys()):  
                self.logger.info("***** Evaluation Result *****")  
                metrics = {}  
                for key, value in logs.items():  
                    self.logger.info(f"  {key}: {value}")  
                    metrics.update({key: value})  
                if "NNI_PLATFORM" in os.environ:  
                    is_final = state.epoch >= args.num_train_epochs  
                    report_nni_metrics(metrics, is_final, self) 
            else:   
                _logs = {k: v for k, v in logs.items() if k not in ["epoch", "step"]}  
                log_str = f"Step {state.global_step} (Epoch {state.epoch:.2f}): " + " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in _logs.items())  
                self.logger.info(log_str)

class DelayedEvaluateEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n_epochs=5, start_epoch=0):
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        self.last_eval_epoch = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(round(state.epoch))
        
        if current_epoch < self.start_epoch:
            control.should_evaluate = False
        
        elif current_epoch % self.n_epochs == 0:
            control.should_evaluate = True
            self.last_eval_epoch = current_epoch
        else:
            control.should_evaluate = False
            
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if int(round(state.epoch)) == self.last_eval_epoch:
             control.should_save = True
