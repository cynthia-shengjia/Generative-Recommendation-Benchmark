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