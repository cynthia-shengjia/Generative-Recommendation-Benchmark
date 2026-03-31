# genrec/trainers/generative/tiger_trainer.py

from typing import Optional, Dict, List, Any, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .base_trainer import BaseGenerativeTrainer
from genrec.generation.trie import Trie, prefix_allowed_tokens_fn
import torch
import math
from transformers import LogitsProcessor
from transformers import LogitsProcessorList
import torch
import math
from transformers import LogitsProcessor

class FastTrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, vocab_size: int):
        self.trie = trie
        self.vocab_size = vocab_size
        self.tensor_mask_cache = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seqs_cpu = input_ids.tolist() 
        
        current_device = scores.device

        for i, seq in enumerate(seqs_cpu):
            seq_tuple = tuple(seq)

            if seq_tuple not in self.tensor_mask_cache:
                allowed_tokens = self.trie.get(seq)
                node_mask = torch.full((self.vocab_size,), -math.inf, device=current_device)
                
                if allowed_tokens:
                    node_mask[allowed_tokens] = 0.0

                self.tensor_mask_cache[seq_tuple] = node_mask
            scores[i, :] += self.tensor_mask_cache[seq_tuple]
        return scores
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
        vocab_size: Optional[int] = None,
        inference_mode: Optional[str] = None
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
            vocab_size=vocab_size,
            inference_mode=inference_mode
        )
        
        # Build Trie for constrained generation
        if self.item2tokens:
            self.candidate_trie = Trie(self.item2tokens)
            if self.inference_mode == "CBS":
                self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
            if self.inference_mode == "FastCBS":
                trie_processor = FastTrieLogitsProcessor(self.candidate_trie,self.vocab_size)
                self.processors = LogitsProcessorList([trie_processor])
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