# genrec/trainers/generative/base_trainer.py

from typing import Optional, Dict, List, Any, Union
from transformers import Trainer, PreTrainedModel
import torch
import torch.nn as nn

class BaseGenerativeTrainer(Trainer):
    """
    Base class for Generative Trainers.
    
    Provides common functionality for generation-based evaluation.
    """
    
    def __init__(
        self,
        model,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        callbacks=None,
        compute_metrics=None,
        generation_params: Optional[Dict] = None,
        item2tokens: Optional[Dict] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        optimizers=(None, None),
    ):
        """
        Initialize Base Generative Trainer.
        
        Args:
            model: The model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            callbacks: List of callbacks
            compute_metrics: Metrics computation function
            generation_params: Generation parameters (max_gen_length, num_beams, max_k)
            item2tokens: Item to tokens mapping
            pad_token_id: Padding token ID
            eos_token_id: EOS token ID
            optimizers: Optimizer and scheduler tuple
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_metrics=compute_metrics,
        )
        
        self.generation_params = generation_params or {}
        self.item2tokens = item2tokens
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        Evaluation step with constrained beam search generation.
        
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
        
        # Prepare inputs
        inputs = self._prepare_inputs(inputs)
        
        # Get labels
        has_labels = "labels" in inputs
        labels = inputs.get("labels")
        
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
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        
        gen_kwargs = {
            "max_length": self.generation_params.get('max_gen_length', 5),
            "num_beams": self.generation_params.get('num_beams', 10),
            "num_return_sequences": self.generation_params.get('max_k', 10),
            "early_stopping": True,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        
        # Add prefix constraint if available
        if self.prefix_allowed_fn:
            gen_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_fn
        
        # Unwrap model for generation
        if hasattr(self, 'accelerator'):
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model
        
        generated_sequences = unwrapped_model.generate(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            **gen_kwargs,
        )
        
        # ===== 3. Reshape results =====
        batch_size = encoder_input_ids.shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        
        # return (loss, generated_ids_reshaped, labels)
        return (loss, generated_ids_reshaped, inputs["label_id"])
