from transformers import TrainerCallback, TrainingArguments, TrainerState
from genrec.utils.nni_utils import report_nni_metrics
import os
# 自定义回调函数来控制评估频率

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


class GenerativeLoggingCallback(TrainerCallback):
    """
    一个自定义的回调函数，将 Trainer 的日志（包括训练进度和评估结果）
    转发到指定的 logger。
    """
    def __init__(self, logger):  
        super().__init__()  
        self.logger = logger  
        self.best_metrics = None  # 添加：跟踪最佳指标
        self.best_score = float('-inf')  # 添加：跟踪最佳分数
  
    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):  
        if state.is_world_process_zero and logs:  
            if any(key.startswith("eval_") for key in logs.keys()):  
                self.logger.info("***** 验证结果 *****")  
                metrics = {}  
                for key, value in logs.items():  
                    self.logger.info(f"  {key}: {value}")  
                    metrics.update({key: value})  
                if "NNI_PLATFORM" in os.environ:  
                    is_final = state.epoch >= args.num_train_epochs  
                    report_nni_metrics(metrics, is_final, self)  # 修改：传递 self
            else:   
                _logs = {k: v for k, v in logs.items() if k not in ["epoch", "step"]}  
                log_str = f"步骤 {state.global_step} (Epoch {state.epoch:.2f}): " + " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in _logs.items())  
                self.logger.info(log_str)

class DelayedEvaluateEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n_epochs=5, start_epoch=0):
        """
        Args:
            n_epochs (int): 每隔多少个 epoch 评估一次
            start_epoch (int): 从第几个 epoch 开始才允许评估
        """
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        self.last_eval_epoch = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch 通常是 float，取整以确保逻辑正确
        current_epoch = int(round(state.epoch))
        
        # 逻辑 1: 如果还没达到起始 epoch，强制关闭评估
        if current_epoch < self.start_epoch:
            control.should_evaluate = False
        
        # 逻辑 2: 达到起始 epoch 后，按间隔评估
        elif current_epoch % self.n_epochs == 0:
            control.should_evaluate = True
            self.last_eval_epoch = current_epoch
        else:
            control.should_evaluate = False
            
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 保持你原有的逻辑：仅在刚刚触发了评估的这个 epoch 保存
        # 注意：state.epoch 在这里也是 float，最好转 int 比较，或者直接设为 True
        # 因为如果进入了 on_evaluate，说明前面的逻辑已经同意评估了
        if int(round(state.epoch)) == self.last_eval_epoch:
             control.should_save = True
>>>>>>> 255d967 (add letter and update sasrec)
