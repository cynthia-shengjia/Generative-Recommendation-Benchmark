# genrec/trainers/online_rl/grpo_trainer.py

from typing import Callable, Optional, Dict, List, Tuple
import torch
from transformers import T5ForConditionalGeneration, TrainerCallback
from accelerate.utils import gather

from .base_trainer import BaseOnlineRLTrainer

class GRPOTrainer(BaseOnlineRLTrainer):
    """
    GRPO (Group Relative Policy Optimization) Trainer.
    
    GRPO uses group-based advantage estimation with optional NDCG rewards.
    """
    
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
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            num_generations=num_generations,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            generation_params=generation_params,
            reward_func=reward_func,
            item2tokens=item2tokens,
            tokens2item=tokens2item,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            optimizers=optimizers,
        )
    
    def _prepare_inputs_for_training(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        GRPO-specific input preparation.
        
        Steps:
        1. Generate sequences using beam search
        2. Optionally add ground truth
        3. Compute rewards
        4. Compute group-based advantages
        5. Get reference model log probabilities
        """
        device = self.accelerator.device
        
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        target_labels = inputs["labels"].to(device)
        
        batch_size = encoder_input_ids.size(0)
        num_beams = self.num_generations
        
        # ===== Generate sequences =====
        if self.add_gt:
            num_gt_per_sample = 1
            num_generated = num_beams - num_gt_per_sample
        else:
            num_gt_per_sample = 0
            num_generated = num_beams
        
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
            generated_ids = outputs.sequences[:, 1:]
            generated_ids = torch.cat([
                generated_ids,
                torch.ones_like(generated_ids[:, :1])
            ], dim=1)
        else:
            generated_ids = None
        
        # ===== Add ground truth =====
        if self.add_gt and num_gt_per_sample > 0:
            gt_decoder_ids = target_labels.repeat_interleave(num_gt_per_sample, dim=0)
        
        # ===== Merge generated and GT =====
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
        
        # ===== Mask after EOS =====
        is_eos = generated_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # ===== Compute rewards =====
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
        
        rewards = self.reward_func(
            generated_items, 
            target_items, 
            num_generations=self.num_generations
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards = gather(rewards)
        
        # ===== Compute advantages =====
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-5)
        
        process_slice = slice(
            self.accelerator.process_index * batch_size * num_beams,
            (self.accelerator.process_index + 1) * batch_size * num_beams,
        )
        advantages = advantages[process_slice]
        sliced_rewards = rewards[process_slice]
        
        # ===== Reference model log probs =====
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
        
        # ===== Log metrics =====
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
        """
        Compute GRPO loss.
        
        Loss = Policy Loss + β * KL Divergence
        """
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        
        encoder_input_ids = inputs["encoder_input_ids"]
        encoder_attention_mask = inputs["encoder_attention_mask"]
        decoder_input_ids = inputs["decoder_input_ids"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]
        
        # Forward pass
        outputs = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            labels=decoder_input_ids,
            return_dict=True,
        )
        logits = outputs.logits
        
        # Compute per-token log probabilities
        labels_clone = decoder_input_ids.clone()
        loss_mask = labels_clone != self.pad_token_id
        labels_clone[labels_clone == self.pad_token_id] = 0
        
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels_clone.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # Compute GRPO loss
        policy_scores = torch.exp(per_token_logps - per_token_logps.detach())
        per_token_loss = policy_scores * advantages.unsqueeze(1)
        
        cross_entropy_loss = -(per_token_loss)
        kl_divergence_loss = per_token_kl
        per_token_loss = cross_entropy_loss + self.beta * kl_divergence_loss
        
        # Average over tokens and batch
        loss = (per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1)
        
        # Log metrics
        mean_kl = ((per_token_kl * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).mean()
        mean_cross_entropy = ((cross_entropy_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)).mean()
        
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl.detach()).mean().item())
        self._metrics["policy_loss"].append(self.accelerator.gather_for_metrics(mean_cross_entropy.detach()).mean().item())
        
        return loss.mean()