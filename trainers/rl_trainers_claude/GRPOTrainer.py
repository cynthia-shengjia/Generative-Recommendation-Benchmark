import os
import math
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Sampler
from accelerate.utils import gather, gather_object, set_seed
from transformers import (
    Trainer,
    TrainerCallback,
    T5ForConditionalGeneration,
    T5Config,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from trl import GRPOConfig
from trl.trainer.utils import pad


if is_wandb_available():
    import wandb

class Trie(object):
    def __init__(self, item2tokens: Dict[int, List] = {}):
        self.trie_dict = {}
        self.len = 0
        sequences = self.add_prefix(item2tokens)
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def add_prefix(self, item2tokens: Dict[int, List]):
        prefix_added_items = [[0] + list(items)  for _, items in item2tokens.items()]
        return prefix_added_items

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)

def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class GRPOTrainerForGenRec(Trainer):
    """
    GRPO Trainer for Generative Recommendation with Encoder-Decoder models.

    Example:
        {
            'input_ids':        [0,0,0,..., 1,178,234,2,169,278] # 代表历史交互序列
            'attention_mask':   [0,0,0,..., 1,1,1,1,1,1,1]  
            'labels':           [5,201,323,1]
        }
    
    Args:
        model: T5ForConditionalGeneration model
        item2tokens: Dict mapping item_id to list of token_ids
        args: GRPOConfig
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        reward_func: Reward function (callable)
        callbacks: List of callbacks
        optimizers: Tuple of optimizer and scheduler
    """

    _tag_names = ["trl", "grpo", "genrec"]

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        item2tokens: Dict[int, List[int]],
        args: GRPOConfig = None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        reward_func: Optional[Callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
    ):
        # Args
        if args is None:
            model_name = model.config._name_or_path if hasattr(model.config, '_name_or_path') else "t5-genrec"
            args = GRPOConfig(f"{model_name}-GRPO")

        # Store item2tokens mapping
        self.item2tokens = item2tokens
        
        # Create reverse mapping: tokens -> item
        self.tokens2item = {}
        for item_id, tokens in item2tokens.items():
            tokens_tuple = tuple(tokens)
            self.tokens2item[tokens_tuple] = item_id

        # Build Trie for constrained generation
        self.candidate_trie = Trie(item2tokens)
        self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)

        # Training arguments
        self.max_seq_len = args.max_prompt_length if args.max_prompt_length else 512
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.beta = args.beta
        self.pad_token_id = model.config.pad_token_id
        self.eos_token_id = model.config.eos_token_id
        self.decoder_start_token_id = model.config.decoder_start_token_id

        # Reward function
        self.reward_func = reward_func if reward_func else self._default_reward_func

        # Create reference model
        self.ref_model = self._create_reference_model(model)

        # Initialize metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions if hasattr(args, 'log_completions') else False

        # Data collator
        if data_collator is None:
            data_collator = lambda features: features

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Set unique seed for each process
        set_seed(args.seed, device_specific=True)

        # Prepare reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                from trl.models import prepare_deepspeed
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # Validation
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )

    def _create_reference_model(self, model):
        """Create a reference model for KL divergence computation."""
        if is_deepspeed_zero3_enabled():
            # Create a new model instance
            ref_model = T5ForConditionalGeneration(model.config)
            ref_model.load_state_dict(model.state_dict())
        else:
            # Create reference model using TRL's utility
            from trl.models import create_reference_model
            ref_model = create_reference_model(model)
        return ref_model

    def _default_reward_func(self, generated_items: List[int], target_items: List[int]) -> List[float]:
        """
        Default reward function: 1.0 if generated item matches target, 0.0 otherwise.
        
        Args:
            generated_items: List of generated item IDs
            target_items: List of target item IDs
            
        Returns:
            List of rewards
        """
        rewards = []
        for gen_item, target_item in zip(generated_items, target_items):
            rewards.append(1.0 if gen_item == target_item else 0.0)
        return rewards

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Expected columns from dataset
            self._signature_columns = ["input_ids", "attention_mask", "labels"]

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        if train_dataset is None:
            train_dataset = self.train_dataset
        return RepeatRandomSampler(train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    def _tokens_to_item(self, token_list: List[int]) -> Optional[int]:
        """Convert a list of tokens to item ID."""
        # Remove padding and special tokens
        clean_tokens = [t for t in token_list if t not in [self.pad_token_id, self.eos_token_id, self.decoder_start_token_id]]
        tokens_tuple = tuple(clean_tokens)
        return self.tokens2item.get(tokens_tuple, None)

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for GRPO training.
        
        This method:
        1. Takes encoder inputs (history items)
        2. Generates multiple completions using beam search
        3. Computes rewards
        4. Computes reference model log probabilities
        """
        """
        Prepare inputs for GRPO training.
        """
        device = self.accelerator.device
        
        # Get encoder inputs
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        target_labels = inputs["labels"].to(device)
        
        batch_size = encoder_input_ids.size(0)
        num_beams = self.num_generations

        
        # Generate completions using beam search with Trie constraint
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                max_length=self.max_completion_length,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                decoder_start_token_id=self.decoder_start_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                prefix_allowed_tokens_fn=self.prefix_allowed_fn,  # 使用包装后的函数
            )

        
        generated_ids = outputs.sequences  # (B * num_beams, gen_len)

        # Mask everything after the first EOS token
        is_eos = generated_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # completion_mask 用来帮助确定哪些token是需要进行 sft 的
        
        # 生成的 response 需要分 group, 然后来计算 group advantage
        
        
        
        # Compute rewards
        generated_items = []
        target_items = []
        
        for i in range(generated_ids.size(0)):
            gen_tokens = generated_ids[i].cpu().tolist()
            gen_item = self._tokens_to_item(gen_tokens)
            generated_items.append(gen_item if gen_item is not None else -1)
            
            # Get corresponding target item
            sample_idx = i // num_beams
            target_tokens = target_labels[sample_idx].cpu().tolist()
            target_item = self._tokens_to_item(target_tokens)
            target_items.append(target_item if target_item is not None else -1)
        

        
        # Compute rewards using reward function
        rewards = self.reward_func(generated_items, target_items)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # Gather rewards across all processes
        rewards = gather(rewards)

        # Compute grouped-wise rewards (normalize within each group)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        
        # Slice to keep only the local part
        process_slice = slice(
            self.accelerator.process_index * batch_size * num_beams,
            (self.accelerator.process_index + 1) * batch_size * num_beams,
        )
        advantages = advantages[process_slice]
        sliced_rewards = rewards[process_slice]
        
        # Compute reference model log probabilities
        with torch.no_grad():
            # Repeat encoder inputs for all generations
            encoder_input_ids_expanded = encoder_input_ids.repeat_interleave(num_beams, dim=0)
            encoder_attention_mask_expanded = encoder_attention_mask.repeat_interleave(num_beams, dim=0)
            
            ref_outputs = self.ref_model(
                input_ids=encoder_input_ids_expanded,
                attention_mask=encoder_attention_mask_expanded,
                decoder_input_ids=generated_ids,
                return_dict=True,
            )
            ref_logits = ref_outputs.logits  # (B * num_beams, gen_len, vocab_size)
            
            # Compute log probabilities
            ref_logits = ref_logits[:, :-1, :]  # Exclude last logit
            generated_ids_for_logp = generated_ids[:, 1:]  # Exclude first token (decoder_start_token)
            
            # Compute log softmax
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            
            # Gather log probs for generated tokens
            ref_per_token_logps = torch.gather(
                ref_log_probs,
                dim=2,
                index=generated_ids_for_logp.unsqueeze(-1)
            ).squeeze(-1)  # (B * num_beams, gen_len - 1)
        
        # Log metrics
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        
        # Compute diversity metrics
        unique_items = len(set([item for item in generated_items if item != -1]))
        total_items = len([item for item in generated_items if item != -1])
        diversity = unique_items / total_items if total_items > 0 else 0.0
        self._metrics["diversity"].append(diversity)
        
        # Compute accuracy
        correct = sum([1 for gen, tgt in zip(generated_items, target_items) if gen == tgt and gen != -1])
        accuracy = correct / len(generated_items) if len(generated_items) > 0 else 0.0
        self._metrics["accuracy"].append(accuracy)
        
        return {
            "encoder_input_ids": encoder_input_ids_expanded,
            "encoder_attention_mask": encoder_attention_mask_expanded,
            "decoder_input_ids": generated_ids,
            "completion_mask": completion_mask[:, 1:],  # Align with logits
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "sliced_rewards": sliced_rewards,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainerForGenRec does not support returning outputs")
        
        encoder_input_ids = inputs["encoder_input_ids"]
        encoder_attention_mask = inputs["encoder_attention_mask"]
        decoder_input_ids = inputs["decoder_input_ids"]
        completion_mask = inputs["completion_mask"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]
        
        # Forward pass through model
        outputs = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        logits = outputs.logits  # (B * num_beams, gen_len, vocab_size)
        
        # Compute log probabilities
        logits = logits[:, :-1, :]  # Exclude last logit
        decoder_input_ids_for_logp = decoder_input_ids[:, 1:]  # Exclude first token
        
        # Compute log softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather log probs for generated tokens
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=decoder_input_ids_for_logp.unsqueeze(-1)
        ).squeeze(-1)  # (B * num_beams, gen_len - 1)
        
        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # Compute GRPO loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        
        # Average over tokens and batch
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[List[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics.clear()