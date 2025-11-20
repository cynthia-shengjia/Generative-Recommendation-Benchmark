import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

class DPOTrainer(Trainer):
    """S-DPO Trainer for Encoder-Decoder models"""
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 0.1,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ):
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.beta = beta
        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=None,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError("Trainer does not have an accelerator object")

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Encoder-Decoder 架构的前向传播
        
        1. 对 chosen 进行前向传播
        2. 对每个 rejected 进行前向传播
        
        返回:
            chosen_logps: [B]
            rejected_logps: [B, N]
            chosen_logits: [B, L, V]
            rejected_logits: [B, N, L, V]
        """
        batch_size = batch["input_ids"].shape[0]
        num_rejected = batch["rejected_labels"].shape[1]
        
        # ===== Forward pass for Chosen =====
        chosen_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["chosen_labels"],
        )
        chosen_logits = chosen_outputs.logits.to(torch.float32)  # [B, L, V]
        
        # 计算 chosen 的 log probabilities
        chosen_logps = self._get_batch_logps(
            chosen_logits,
            batch["chosen_labels"],
            average_log_prob=False,
        )  # [B]
        
        # ===== Forward pass for Rejected (多个) =====
        # 方法：循环处理每个 rejected，或者 batch 处理
        # 这里使用 batch 处理：将 [B, N, L] reshape 成 [B*N, L]
        
        # rejected_labels: [B, N, L] -> [B*N, L]
        rejected_labels_flat = batch["rejected_labels"].view(batch_size * num_rejected, -1)
        
        # 复制 input_ids 和 attention_mask N 次
        # input_ids: [B, L] -> [B*N, L]
        input_ids_repeated = batch["input_ids"].repeat_interleave(num_rejected, dim=0)
        attention_mask_repeated = batch["attention_mask"].repeat_interleave(num_rejected, dim=0)
        
        # Forward pass
        rejected_outputs = model(
            input_ids=input_ids_repeated,
            attention_mask=attention_mask_repeated,
            labels=rejected_labels_flat,
        )
        rejected_logits_flat = rejected_outputs.logits.to(torch.float32)  # [B*N, L, V]
        
        # 计算 rejected 的 log probabilities
        rejected_logps_flat = self._get_batch_logps(
            rejected_logits_flat,
            rejected_labels_flat,
            average_log_prob=False,
        )  # [B*N]
        
        # Reshape: [B*N] -> [B, N]
        rejected_logps = rejected_logps_flat.view(batch_size, num_rejected)
        
        # Reshape logits: [B*N, L, V] -> [B, N, L, V]
        rejected_logits = rejected_logits_flat.view(
            batch_size, num_rejected, rejected_logits_flat.shape[1], rejected_logits_flat.shape[2]
        )
        
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,      # [B]
        policy_rejected_logps: torch.FloatTensor,    # [B, N]
        reference_chosen_logps: torch.FloatTensor,   # [B]
        reference_rejected_logps: torch.FloatTensor, # [B, N]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        S-DPO Loss: 同时对比 chosen 和多个 rejected
        
        Loss = -log σ(-log Σ_i exp(β * (r_rejected_i - r_chosen)))
        
        Args:
            policy_chosen_logps: [B]
            policy_rejected_logps: [B, N]
            reference_chosen_logps: [B]
            reference_rejected_logps: [B, N]
        
        Returns:
            losses: [B]
            chosen_rewards: [B]
            rejected_rewards: [B, N]
        """
        # 计算 log-ratios
        chosen_logratios = policy_chosen_logps - reference_chosen_logps  # [B]
        rejected_logratios = policy_rejected_logps - reference_rejected_logps  # [B, N]
        
        # S-DPO 核心：对所有 rejected 做 softmax
        # chosen_logratios: [B] -> [B, 1] for broadcasting
        chosen_logratios_expanded = chosen_logratios.unsqueeze(1)  # [B, 1]
        
        # rejected_logratios - chosen_logratios: [B, N] - [B, 1] -> [B, N]
        logit_diff = rejected_logratios - chosen_logratios_expanded  # [B, N]
        
        # temp = Σ_i exp(β * (r_rejected_i - r_chosen))
        # sum over N dimension: [B, N] -> [B]
        temp = torch.sum(torch.exp(self.beta * logit_diff), dim=1)  # [B]
        
        # Loss = -log σ(-log temp)
        temp1 = -torch.log(temp)  # [B]
        losses = -F.logsigmoid(temp1)  # [B]
        
        # 计算 rewards（用于 logging）
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()  # [B]
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()  # [B, N]
        
        return losses, chosen_rewards, rejected_rewards

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """计算 batch 的 loss 和 metrics"""
        metrics = {}
        
        # Policy model 前向传播
        (
            policy_chosen_logps,      # [B]
            policy_rejected_logps,    # [B, N]
            policy_chosen_logits,     # [B, L, V]
            policy_rejected_logits,   # [B, N, L, V]
        ) = self.concatenated_forward(model, batch)
        
        # Reference model 前向传播
        with torch.no_grad():
            (
                reference_chosen_logps,    # [B]
                reference_rejected_logps,  # [B, N]
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch)
        
        # 计算 loss
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        # 计算 accuracy: chosen 是否比所有 rejected 都好
        # chosen_rewards: [B] -> [B, 1]
        # rejected_rewards: [B, N]
        # chosen_rewards > rejected_rewards: [B, N]
        reward_comparisons = chosen_rewards.unsqueeze(1) > rejected_rewards  # [B, N]
        # 所有 N 个 rejected 都要小于 chosen
        reward_accuracies = reward_comparisons.all(dim=1).float()  # [B]
        
        # Logging metrics
        prefix = "eval_" if train_eval == "eval" else ""
        
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/rejected_mean"] = rejected_rewards.cpu().numpy().mean()
        
        # 每个 rejected 的平均 reward
        num_rejected = rejected_rewards.shape[1]
        for i in range(num_rejected):
            metrics[f"{prefix}rewards/rejected{i+1}"] = rejected_rewards[:, i].cpu().numpy().mean()
        
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        
        # Margins: chosen - rejected
        margins = chosen_rewards.unsqueeze(1) - rejected_rewards  # [B, N]
        metrics[f"{prefix}rewards/margins_mean"] = margins.cpu().numpy().mean()
        for i in range(num_rejected):
            metrics[f"{prefix}rewards/margins_rejected{i+1}"] = margins[:, i].cpu().numpy().mean()
        
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/rejected_mean"] = policy_rejected_logps.detach().cpu().numpy().mean()
        
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/rejected_mean"] = policy_rejected_logits.detach().cpu().numpy().mean()
        
        return losses.mean(), metrics

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """
        计算给定 labels 在 logits 下的 log probabilities
        
        Args:
            logits: [B, L, V]
            labels: [B, L]
        
        Returns:
            log_probs: [B]
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape (except last dim)")
        
        # Shift: 预测下一个 token
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        
        # Mask: 忽略 label_pad_token_id
        loss_mask = labels != self.label_pad_token_id
        
        # 将 pad token 替换为 0（避免索引错误）
        labels[labels == self.label_pad_token_id] = 0
        
        # 计算每个 token 的 log probability
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """训练时调用"""
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        
        # Log metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return (loss, metrics)
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """评估时调用"""
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")
        
        # Log metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")
        
        if prediction_loss_only:
            return (loss.detach(), None, None)
        
        # 返回 logits 和 labels（用于计算其他指标）
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1) if logits else torch.tensor([0.0])
        labels = torch.zeros(logits.shape[0])
        
        return (loss.detach(), logits, labels)

    def store_metrics(
        self,
        metrics: Dict[str, float],
        train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        """存储 metrics"""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log metrics to the various objects watching training.
        
        Args:
            logs: Dictionary of metrics to log
            *args, **kwargs: Additional arguments passed to parent's log method
        """
        train_eval = "train" if "loss" in logs else "eval"
        
        # Add stored metrics to logs
        if train_eval in self._stored_metrics:
            for key, metrics in self._stored_metrics[train_eval].items():
                if len(metrics) > 0:
                    logs[key] = torch.tensor(metrics).mean().item()
            
            # Clear stored metrics for this phase
            self._stored_metrics[train_eval].clear()
        
        # Call parent's log method
        return super().log(logs, *args, **kwargs)