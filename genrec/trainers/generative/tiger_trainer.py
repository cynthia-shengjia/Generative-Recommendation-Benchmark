# genrec/trainers/generative/tiger_trainer.py

from typing import Optional, Dict, List, Any, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .base_trainer import BaseGenerativeTrainer
from genrec.generation.trie import Trie, prefix_allowed_tokens_fn

class TigerTrainer(BaseGenerativeTrainer):
    """
    Tiger Trainer for Generative Recommendation.
    
    Supports constrained beam search generation during evaluation.
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
        Initialize Tiger Trainer.
        
        Args:
            model: T5 model for generation
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            callbacks: List of callbacks
            compute_metrics: Metrics computation function
            generation_params: Generation parameters (max_gen_length, num_beams, max_k)
            item2tokens: Item to tokens mapping (for constrained generation)
            pad_token_id: Padding token ID
            eos_token_id: EOS token ID
            optimizers: Optimizer and scheduler tuple
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            generation_params=generation_params,
            item2tokens=item2tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            optimizers=optimizers,
        )
        
        # Build Trie for constrained generation
        if self.item2tokens:
            self.candidate_trie = Trie(self.item2tokens)
            self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        else:
            self.candidate_trie = None
            self.prefix_allowed_fn = None
    
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        
        loss_mask = inputs.pop("loss_mask", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits 
        loss = None
        if labels is not None:
            #loss_mask [batch_size, seq_len, vocab_size]
            if loss_mask is not None:
                
                masked_logits = logits.masked_fill(loss_mask == 0.0, -1e9)

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                unwrapped_model = self.accelerator.unwrap_model(model)
                loss = loss_fct(
                    masked_logits.view(-1, unwrapped_model.config.vocab_size),
                    labels.view(-1)
                )
            else:
                loss = outputs.loss

        return (loss, outputs) if return_outputs else loss