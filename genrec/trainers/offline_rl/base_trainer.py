# genrec/trainers/offline_rl/base_trainer.py

from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_outputs import BaseModelOutput

from genrec.generation.trie import Trie, prefix_allowed_tokens_fn

class BaseOfflineRLTrainer(Trainer, ABC):
    """
    Base Trainer for Offline RL methods in Generative Recommendation.
    
    This base class handles common functionality:
    - Reference model management
    - Trie-based constrained generation
    - Evaluation with generation
    - Metrics logging
    
    Subclasses need to implement:
    - concatenated_forward: Forward pass for policy and reference models
    - compute_rl_loss: Algorithm-specific loss computation
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 0.1,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        eval_data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # Evaluation parameters
        compute_metrics: Optional[Callable] = None,
        generation_params: Optional[Dict] = None,
        item2tokens: Optional[Dict] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Base Offline RL Trainer.
        
        Args:
            model: Policy model
            ref_model: Reference model (frozen)
            beta: KL penalty coefficient
            args: Training arguments
            data_collator: Training data collator
            eval_data_collator: Evaluation data collator
            label_pad_token_id: Label padding token ID
            padding_value: Padding value
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            model_init: Model initialization function
            callbacks: List of callbacks
            optimizers: Optimizer and scheduler tuple
            preprocess_logits_for_metrics: Logits preprocessing function
            compute_metrics: Metrics computation function
            generation_params: Generation parameters (max_gen_length, num_beams, max_k)
            item2tokens: Item to tokens mapping
            pad_token_id: Padding token ID
            eos_token_id: EOS token ID
        """
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.beta = beta
        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.eval_data_collator = eval_data_collator
        
        # Evaluation parameters
        self.generation_params = generation_params or {}
        self.item2tokens = item2tokens
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.eos_token_id = eos_token_id if eos_token_id is not None else 1
        
        # Build Trie for constrained generation
        if self.item2tokens:
            self.candidate_trie = Trie(self.item2tokens)
            self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        else:
            self.prefix_allowed_fn = None
            print("⚠️ 警告: 未提供 item2tokens，无法使用前缀约束生成。")
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError("Trainer does not have an accelerator object")
    
    @abstractmethod
    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple:
        """
        Algorithm-specific forward pass.
        
        This method should be implemented by subclasses to define how to:
        1. Process chosen and rejected samples
        2. Compute log probabilities
        3. Return necessary outputs for loss computation
        
        Args:
            model: The model (policy or reference)
            batch: Input batch
        
        Returns:
            Tuple of outputs (algorithm-specific)
        """
        raise NotImplementedError("Subclasses must implement concatenated_forward")
    
    @abstractmethod
    def compute_rl_loss(
        self,
        policy_outputs: Tuple,
        reference_outputs: Tuple,
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Algorithm-specific loss computation.
        
        This method should be implemented by subclasses.
        
        Args:
            policy_outputs: Outputs from policy model's concatenated_forward
            reference_outputs: Outputs from reference model's concatenated_forward
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        raise NotImplementedError("Subclasses must implement compute_rl_loss")
    
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """
        Compute log probabilities for given labels under logits.
        
        Args:
            logits: Model logits [B, L, V] or [B*N, L, V]
            labels: Target labels [B, L] or [B*N, L]
            average_log_prob: Whether to average log probs over sequence length
        
        Returns:
            Log probabilities [B] or [B*N]
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape (except last dim)")
        
        labels_clone = labels.clone()
        loss_mask = labels_clone != self.label_pad_token_id
        labels_clone[labels_clone == self.label_pad_token_id] = 0
        
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels_clone.unsqueeze(2)
        ).squeeze(2)
        
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
    
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Compute batch loss and metrics.
        
        Args:
            model: The model
            batch: Input batch
            train_eval: "train" or "eval"
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Policy model forward
        policy_outputs = self.concatenated_forward(model, batch)
        
        # Reference model forward
        with torch.no_grad():
            reference_outputs = self.concatenated_forward(self.ref_model, batch)
        
        # Compute loss
        loss, metrics = self.compute_rl_loss(policy_outputs, reference_outputs)
        
        # Add prefix for eval metrics
        if train_eval == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        
        return loss, metrics
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Training loss computation"""
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        
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
        """
        Evaluation step with generation.
        
        Args:
            model: The model
            inputs: Input dictionary
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore
        
        Returns:
            Tuple of (loss, predictions, labels)
        """
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        inputs = self._prepare_inputs(inputs)
        
        # Get labels (prefer chosen_labels)
        has_labels = "chosen_labels" in inputs or "labels" in inputs
        labels = inputs.get("chosen_labels", inputs.get("labels"))
        
        # ===== 1. Compute loss =====
        with torch.no_grad():
            if has_labels:
                loss_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "labels": labels,
                }
                outputs = model(**loss_inputs)
                loss = outputs.loss.mean().detach() if outputs.loss is not None else torch.tensor(0.0)
            else:
                loss = torch.tensor(0.0)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # ===== 2. Generate sequences =====
        device = self.accelerator.device if hasattr(self, 'accelerator') else next(model.parameters()).device
        
        gen_kwargs = {
            "max_length": self.generation_params.get('max_gen_length', 5),
            "num_beams": self.generation_params.get('num_beams', 10),
            "num_return_sequences": self.generation_params.get('max_k', 10),
            "early_stopping": True,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        
        if self.prefix_allowed_fn:
            gen_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_fn
        
        unwrapped_model = self.accelerator.unwrap_model(model) if hasattr(self, 'accelerator') else model
        generated_sequences = unwrapped_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        
        # ===== 3. Reshape results =====
        batch_size = inputs["input_ids"].shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        
        return (loss, generated_ids_reshaped, labels)
    
    def store_metrics(
        self,
        metrics: Dict[str, float],
        train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        """Store metrics"""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Log metrics"""
        train_eval = "train" if "loss" in logs else "eval"
        
        if train_eval in self._stored_metrics:
            for key, metrics in self._stored_metrics[train_eval].items():
                if len(metrics) > 0:
                    logs[key] = torch.tensor(metrics).mean().item()
            
            self._stored_metrics[train_eval].clear()
        
        return super().log(logs, *args, **kwargs)
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use different collator for evaluation"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        original_collator = self.data_collator
        if self.eval_data_collator is not None:
            self.data_collator = self.eval_data_collator
        
        dataloader = super().get_eval_dataloader(eval_dataset)
        
        self.data_collator = original_collator
        
        return dataloader