# genrec/trainers/online_rl/base_trainer.py

from collections import defaultdict
from typing import Any, Callable, Optional, Union, Dict, List, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from accelerate.utils import gather
from transformers import (
    Trainer,
    TrainerCallback,
    T5ForConditionalGeneration,
    PreTrainedModel,
)

from genrec.generation.trie import Trie, prefix_allowed_tokens_fn

class BaseOnlineRLTrainer(Trainer, ABC):
    """
    Base Trainer for Online RL methods in Generative Recommendation.
    
    This base class handles common functionality:
    - Tokenizer and Trie setup
    - Reference model management
    - Evaluation with generation
    - Metrics logging
    - Distributed training utilities (gather, compute, slice)
    
    Subclasses need to implement:
    - _prepare_inputs_for_training: Training-specific input preparation
    - compute_loss: Algorithm-specific loss computation
    """
    
    _tag_names = ["trl", "online_rl", "genrec"]
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        ref_model: T5ForConditionalGeneration,
        beta: float,
        num_generations: int,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        callbacks: Optional[List[TrainerCallback]] = None,
        compute_metrics: Optional[Callable] = None,
        generation_params: Optional[Dict] = None,
        reward_func: Optional[Callable] = None,
        item2tokens: Optional[Dict] = None,
        tokens2item: Optional[Dict] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
    ):
        # ... (same as before)
        
        # ===== Tokenizer and Trie =====
        self.item2tokens = item2tokens
        self.tokens2item = tokens2item
        self.candidate_trie = Trie(self.item2tokens)
        self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        
        # ===== Training parameters =====
        self.num_generations = num_generations
        self.beta = beta
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = model.config.decoder_start_token_id
        
        # ===== Generation parameters =====
        self.generation_params = generation_params or {}
        self.max_completion_length = self.generation_params.get('max_gen_length', 5)
        
        # ===== Reward function =====
        self.reward_func = reward_func if reward_func else self._default_reward_func
        
        # ===== Reference model =====
        self.ref_model = ref_model
        
        # ===== Metrics =====
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions if hasattr(args, 'log_completions') else False
        
        # ===== Training flags =====
        self.add_gt = True
        
        # ===== Initialize parent Trainer =====
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_metrics=compute_metrics
        )
        
        # ===== Prepare reference model with accelerator =====
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError("Trainer does not have an accelerator object")
    
    def _default_reward_func(self, generated_items: List[int], target_items: List[int], **kwargs) -> List[float]:
        """Default reward function: 1.0 if match, 0.0 otherwise."""
        rewards = []
        for gen_item, target_item in zip(generated_items, target_items):
            rewards.append(1.0 if gen_item == target_item else 0.0)
        return rewards
    
    def _tokens_to_item(self, token_list: List[int]) -> Optional[int]:
        """Convert a list of tokens to item ID."""
        clean_tokens = [
            t for t in token_list 
            if t not in [self.pad_token_id, self.eos_token_id, self.decoder_start_token_id]
        ]
        tokens_tuple = tuple(clean_tokens)
        return self.tokens2item.get(tokens_tuple, None)
    
    def _gather_compute_slice(
        self,
        tensors_to_gather: Dict[str, torch.Tensor],
        batch_size: int,
        num_seqs_per_sample: int,
        compute_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        return_gathered: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Universal gather-compute-slice pattern for distributed training.
        
        This method abstracts the common pattern:
        1. Gather tensors from all processes
        2. Compute advantages/values on gathered data
        3. Slice back to current process
        
        Args:
            tensors_to_gather: Dict of tensors to gather, each with shape (batch_size * num_seqs_per_sample,) or (batch_size,)
            batch_size: Batch size per process
            num_seqs_per_sample: Number of sequences per sample
            compute_fn: Function to compute advantages from gathered tensors
                       Input: Dict of gathered tensors (same keys as tensors_to_gather)
                       Output: Computed advantages, shape (total_samples * num_seqs_per_sample,)
            return_gathered: Whether to return gathered tensors (for logging)
        
        Returns:
            Tuple of:
            - advantages: Computed advantages for current process, shape (batch_size * num_seqs_per_sample,)
            - gathered_tensors: Dict of gathered tensors (if return_gathered=True), otherwise empty dict
        
        Example usage for GRPO:
            advantages, gathered = self._gather_compute_slice(
                tensors_to_gather={"rewards": rewards},
                batch_size=batch_size,
                num_seqs_per_sample=num_beams,
                compute_fn=lambda g: (g["rewards"].view(-1, num_beams) - g["rewards"].view(-1, num_beams).mean(1, keepdim=True)) 
                                     / (g["rewards"].view(-1, num_beams).std(1, keepdim=True) + 1e-5)).view(-1),
            )
        
        Example usage for RankPO:
            advantages, gathered = self._gather_compute_slice(
                tensors_to_gather={"scores": scores, "is_positive": is_positive, "quantiles": quantiles},
                batch_size=batch_size,
                num_seqs_per_sample=num_seqs,
                compute_fn=lambda g: compute_rankpo_advantages(g, num_seqs, tau),
            )
        """
        # ===== Step 1: Gather tensors from all processes =====
        gathered_tensors = {}
        for key, tensor in tensors_to_gather.items():
            gathered_tensors[key] = gather(tensor)
        
        # ===== Step 2: Compute advantages on gathered data =====
        advantages = compute_fn(gathered_tensors)
        
        # ===== Step 3: Slice back to current process =====
        process_slice = slice(
            self.accelerator.process_index * batch_size * num_seqs_per_sample,
            (self.accelerator.process_index + 1) * batch_size * num_seqs_per_sample,
        )
        sliced_advantages = advantages[process_slice]
        
        # ===== Step 4: Optionally slice gathered tensors for logging =====
        if return_gathered:
            sliced_gathered = {key: tensor[process_slice] for key, tensor in gathered_tensors.items()}
            return sliced_advantages, {**gathered_tensors, **{f"sliced_{k}": v for k, v in sliced_gathered.items()}}
        else:
            return sliced_advantages, {}
    
    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for training or evaluation.
        
        - Training mode: Use algorithm-specific preparation
        - Evaluation mode: Use standard preparation
        """
        if not self.model.training:
            return super()._prepare_inputs(inputs)
        return self._prepare_inputs_for_training(inputs)
    
    @abstractmethod
    def _prepare_inputs_for_training(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Algorithm-specific input preparation for training.
        
        This method should be implemented by subclasses to define how to:
        1. Generate sequences
        2. Compute rewards/advantages
        3. Prepare reference model outputs
        """
        raise NotImplementedError("Subclasses must implement _prepare_inputs_for_training")
    
    @abstractmethod
    def compute_loss(
        self, 
        model, 
        inputs, 
        return_outputs=False, 
        num_items_in_batch=None
    ):
        """Compute algorithm-specific loss."""
        raise NotImplementedError("Subclasses must implement compute_loss")
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Evaluation step with generation."""
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        inputs = self._prepare_inputs(inputs)
        has_labels = "labels" in inputs
        labels = inputs.get("labels")
        
        # Compute loss
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
        
        # Generate sequences
        device = self.accelerator.device
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        
        gen_kwargs = {
            "max_length": self.generation_params.get('max_gen_length', 5),
            "num_beams": self.generation_params.get('num_beams', 10),
            "num_return_sequences": self.generation_params.get('max_k', 10),
            "early_stopping": True,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
        }
        
        if hasattr(self, 'prefix_allowed_fn') and self.prefix_allowed_fn:
            gen_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_fn
        
        unwrapped_model = self.accelerator.unwrap_model(model)
        generated_sequences = unwrapped_model.generate(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            **gen_kwargs,
        )
        
        batch_size = encoder_input_ids.shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        
        return (loss, generated_ids_reshaped, labels)
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics."""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items() if len(val) > 0}
        
        if logs and next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics.clear()