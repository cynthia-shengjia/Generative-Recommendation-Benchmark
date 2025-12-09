
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Dict, List, Tuple

import torch.nn as nn    

import torch
from torch.utils.data import Sampler
from accelerate.utils import gather
from transformers import (
    Trainer,
    TrainerCallback,
    T5ForConditionalGeneration,
)

from transformers import PreTrainedModel, Trainer    
from genrec.generation.trie import Trie,prefix_allowed_tokens_fn



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


class GRPOTrainer(Trainer):
    """
    GRPO Trainer for Generative Recommendation with Encoder-Decoder models.
    """
    
    _tag_names = ["trl", "grpo", "genrec"]
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        ref_model,
        beta,
        num_generations,
        args = None,
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
        
        # Get item2tokens from tokenizer
        self.item2tokens = item2tokens
        self.tokens2item = tokens2item
        
        
        # Build Trie for constrained generation
        self.candidate_trie = Trie(self.item2tokens)
        self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        
        # Training arguments

        
        self.num_generations = num_generations
        
        self.beta = beta
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = model.config.decoder_start_token_id
        self.generation_params = generation_params or {}  
        self.max_completion_length = self.generation_params.get('max_gen_length',5)

        # Reward function
        self.reward_func = reward_func if reward_func else self._default_reward_func
        

        self.ref_model = ref_model
        
        # Initialize metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions if hasattr(args, 'log_completions') else False
        self.add_gt = True
        
        
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
                

        if hasattr(self, "accelerator"):  
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)  
        else:  
            raise AttributeError("Trainer does not have an accelerator object")  
      

        # Validation
        # num_processes = self.accelerator.num_processes
        # global_batch_size = args.per_device_train_batch_size * num_processes
        # possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )


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

    def _tokens_to_item(self, token_list: List[int]) -> Optional[int]:
        """Convert a list of tokens to item ID."""
        # Remove padding and special tokens
        clean_tokens = [t for t in token_list if t not in [self.pad_token_id, self.eos_token_id, self.decoder_start_token_id]]
        tokens_tuple = tuple(clean_tokens)
        return self.tokens2item.get(tokens_tuple, None)
    
    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        准备输入数据。
        - 训练时：使用 GRPO 的完整逻辑
        - 评估时：使用标准的输入准备
        """
        # 🔴 关键修改：检查是否在评估模式
        if not self.model.training:
            # 评估模式：直接返回标准输入（调用父类方法）
            return super()._prepare_inputs(inputs)
        
        # 训练模式：使用 GRPO 的完整逻辑
        return self._prepare_inputs_for_grpo(inputs)
    
    def _prepare_inputs_for_grpo(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        GRPO 训练专用的输入准备（原来的 _prepare_inputs 逻辑）
        """
        device = self.accelerator.device
        
        # Get encoder inputs
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        target_labels = inputs["labels"].to(device)
        
        batch_size = encoder_input_ids.size(0)
        num_beams = self.num_generations
        
        # Calculate how many samples to generate vs how many GT to add
        if self.add_gt:
            num_gt_per_sample = 1
            num_generated = num_beams - num_gt_per_sample
        else:
            num_gt_per_sample = 0
            num_generated = num_beams
        
        # ========== Part 1: Generate completions ==========
        if num_generated > 0:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    max_length=self.max_completion_length,
                    num_beams=num_generated,
                    num_return_sequences=num_generated,
                    early_stopping=True,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    decoder_start_token_id=self.decoder_start_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    prefix_allowed_tokens_fn=self.prefix_allowed_fn,
                )
            generated_ids = outputs.sequences
        else:
            generated_ids = None
        
        # ========== Part 2: Add GT samples ==========
        if self.add_gt and num_gt_per_sample > 0:
            gt_decoder_ids = target_labels.repeat_interleave(num_gt_per_sample, dim=0)
        
        # ========== Part 3: Merge generated and GT samples ==========
        if self.add_gt and num_gt_per_sample > 0:
            if generated_ids is not None:
                max_len = max(generated_ids.size(1), gt_decoder_ids.size(1))
                
                if generated_ids.size(1) < max_len:
                    padding = torch.full(
                        (generated_ids.size(0), max_len - generated_ids.size(1)),
                        self.pad_token_id,
                        dtype=generated_ids.dtype,
                        device=device
                    )
                    generated_ids = torch.cat([generated_ids, padding], dim=1)
                
                if gt_decoder_ids.size(1) < max_len:
                    padding = torch.full(
                        (gt_decoder_ids.size(0), max_len - gt_decoder_ids.size(1)),
                        self.pad_token_id,
                        dtype=gt_decoder_ids.dtype,
                        device=device
                    )
                    gt_decoder_ids = torch.cat([gt_decoder_ids, padding], dim=1)
                
                generated_ids_reshaped = generated_ids.view(batch_size, num_generated, max_len)
                gt_decoder_ids_reshaped = gt_decoder_ids.view(batch_size, num_gt_per_sample, max_len)
                all_decoder_ids = torch.cat([generated_ids_reshaped, gt_decoder_ids_reshaped], dim=1)
                all_decoder_ids = all_decoder_ids.view(batch_size * num_beams, max_len)
            else:
                all_decoder_ids = gt_decoder_ids
        else:
            all_decoder_ids = generated_ids
        
        generated_ids = all_decoder_ids
        
        # ========== Part 4: Mask after EOS ==========
        is_eos = generated_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # ========== Part 5: Compute rewards ==========
        generated_items = []
        target_items = []
        
        for i in range(generated_ids.size(0)):
            gen_tokens = generated_ids[i].cpu().tolist()
            gen_item = self._tokens_to_item(gen_tokens)
            generated_items.append(gen_item if gen_item is not None else -1)
            
            sample_idx = i // num_beams
            target_tokens = target_labels[sample_idx].cpu().tolist()
            target_item = self._tokens_to_item(target_tokens)
            target_items.append(target_item if target_item is not None else -1)
        
        rewards = self.reward_func(generated_items, target_items, num_generations=self.num_generations)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards = gather(rewards)
        
        # ========== Part 6: Compute advantages ==========
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-5)
        # advantages = rewards
        
        process_slice = slice(
            self.accelerator.process_index * batch_size * num_beams,
            (self.accelerator.process_index + 1) * batch_size * num_beams,
        )
        advantages = advantages[process_slice]
        sliced_rewards = rewards[process_slice]
        
        # ========== Part 7: Compute reference log probs ==========

        with torch.no_grad():
            encoder_input_ids_expanded = encoder_input_ids.repeat_interleave(num_beams, dim=0)
            encoder_attention_mask_expanded = encoder_attention_mask.repeat_interleave(num_beams, dim=0)
            
            ref_outputs = self.ref_model(
                input_ids=encoder_input_ids_expanded,
                attention_mask=encoder_attention_mask_expanded,
                labels=generated_ids,
                return_dict=True,
            )
            ref_logits = ref_outputs.logits

            ref_per_token_logps = torch.gather(
                ref_logits.log_softmax(-1),
                dim=2,
                index=generated_ids.unsqueeze(-1)
            ).squeeze(-1)
        
        # ========== Part 8: Log metrics ==========
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        
        if self.add_gt and num_gt_per_sample > 0:
            rewards_reshaped = rewards.view(batch_size, num_beams)
            gt_rewards = rewards_reshaped[:, -num_gt_per_sample:].mean()
            self._metrics["gt_reward"].append(gt_rewards.item())
            
            if num_generated > 0:
                gen_rewards = rewards_reshaped[:, :num_generated].mean()
                self._metrics["gen_reward"].append(gen_rewards.item())
        
        unique_items = len(set([item for item in generated_items if item != -1]))
        total_items = len([item for item in generated_items if item != -1])
        diversity = unique_items / total_items if total_items > 0 else 0.0
        self._metrics["diversity"].append(diversity)
        
        correct = sum([1 for gen, tgt in zip(generated_items, target_items) if gen == tgt and gen != -1])
        accuracy = correct / len(generated_items) if len(generated_items) > 0 else 0.0
        self._metrics["accuracy"].append(accuracy)
        
        return {
            "encoder_input_ids": encoder_input_ids_expanded,
            "encoder_attention_mask": encoder_attention_mask_expanded,
            "decoder_input_ids": generated_ids,
            "completion_mask": completion_mask,
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

        # Forward pass
        outputs = model(
            input_ids=encoder_input_ids, 
            attention_mask=encoder_attention_mask, 
            labels=decoder_input_ids,  # ✅ 只传 decoder_input_ids
            return_dict=True, 
        )
        logits = outputs.logits  # [B*num_beams, L, vocab_size]
        
        
        shifted_labels = decoder_input_ids    
        shifted_logits = logits    

        labels_clone = shifted_labels.clone()
        loss_mask = labels_clone != self.pad_token_id
        labels_clone[labels_clone == self.pad_token_id] = 0
  

        per_token_logps = torch.gather( 
            shifted_logits.log_softmax(-1),    
            dim=2, 
            index=labels_clone.unsqueeze(-1)
        ).squeeze(-1)


        # Cross-entropy: loss = -(per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)    

        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # # Compute GRPO loss
        policy_scores  = torch.exp(per_token_logps - per_token_logps.detach()) 
        per_token_loss = policy_scores * advantages.unsqueeze(1)
        
        cross_entropy_loss = -(per_token_loss)
        kl_divergence_loss = per_token_kl
        per_token_loss     = cross_entropy_loss + self.beta * kl_divergence_loss
        
        # # Average over tokens and batch
        # # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1)    

        # # Log metrics
        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        # self._metrics["completion_length"].append(completion_length)
        
        mean_kl             = ((per_token_kl * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).mean()
        mean_cross_entropy  = ((cross_entropy_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).mean()
        
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl.detach()).mean().item())
        self._metrics["policy_loss"].append(self.accelerator.gather_for_metrics(mean_cross_entropy.detach()).mean().item())

        return loss.mean()

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        评估时调用 - 使用生成式评估
        """
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        # ===== 准备输入 =====
        inputs = self._prepare_inputs(inputs)
        
        # 获取 labels
        has_labels = "labels" in inputs
        labels = inputs.get("labels")
        
        # ===== 1. 计算损失（使用 GRPO 的 _prepare_inputs 和 compute_loss）=====
        with torch.no_grad():
            if has_labels:
                # 使用 GRPO 的完整流程计算 loss
                loss_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "labels": labels,
                }
                outputs = model(**loss_inputs)
                loss = outputs.loss.mean().detach() if outputs.loss is not None else torch.tensor(0.0)
            else:
                loss = torch.tensor(0.0)
        
        # 如果只需要 loss，直接返回
        if prediction_loss_only:
            return (loss, None, None)
        
        # ===== 2. 执行生成操作（用于评估指标）=====
        device = self.accelerator.device
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        
        # 生成参数
        gen_kwargs = {
            "max_length": self.generation_params.get('max_gen_length', 5),
            "num_beams": self.generation_params.get('num_beams', 10),
            "num_return_sequences": self.generation_params.get('max_k', 10),
            "early_stopping": True,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
        }
        
        # 🔴 添加前缀约束（使用 GRPO 的 Trie）
        if hasattr(self, 'prefix_allowed_fn') and self.prefix_allowed_fn:
            gen_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_fn
        
        # 执行生成
        unwrapped_model = self.accelerator.unwrap_model(model)
        generated_sequences = unwrapped_model.generate(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            **gen_kwargs,
        )
        
        # ===== 3. Reshape 生成结果 =====
        # (batch_size * num_beams, seq_len) -> (batch_size, num_beams, seq_len)
        batch_size = encoder_input_ids.shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        
        # ===== 4. 返回结果 =====
        # (loss, predictions, labels)
        # predictions: 生成的序列 [B, num_beams, L]
        # labels: 原始 labels（用于 compute_metrics）
        return (loss, generated_ids_reshaped, labels)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics.clear()