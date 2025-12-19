# genrec/trainers/online_rl/rankpo_trainer.py

from typing import Callable, Optional, Dict, List, Tuple
import torch
from transformers import T5ForConditionalGeneration, TrainerCallback
from accelerate.utils import gather

from .base_trainer import BaseOnlineRLTrainer

class RankPOTrainer(BaseOnlineRLTrainer):
    """
    RankPO (Ranking Policy Optimization) Trainer.
    
    RankPO uses quantile-based advantage estimation without explicit rewards.
    """
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        ref_model: T5ForConditionalGeneration,
        beta: float,
        tau: float,  # 🔥 RankPO-specific parameter
        num_generations: int,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        callbacks: Optional[List[TrainerCallback]] = None,
        compute_metrics: Optional[Callable] = None,
        generation_params: Optional[Dict] = None,
        reward_func: Optional[Callable] = None,  # RankPO doesn't use reward_func
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
        
        # RankPO-specific parameter
        self.tau = tau
    
    def _prepare_inputs_for_training(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        RankPO-specific input preparation.
        
        Steps:
        1. Generate sequences and get their scores
        2. Compute quantile from scores
        3. Optionally add ground truth with its score
        4. Compute advantages based on quantile and positive/negative labels
        5. Get reference model log probabilities
        """
        device = self.accelerator.device
        
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        target_labels = inputs["labels"].to(device)
        
        batch_size = encoder_input_ids.size(0)
        num_beams = self.num_generations
        
        # ===== Generate sequences with scores =====
        if self.add_gt:
            num_generated = num_beams - 1
        else:
            num_generated = num_beams
        
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
                torch.full_like(generated_ids[:, :1], self.eos_token_id)
            ], dim=1)
            generated_scores = outputs.sequences_scores
        
        seq_len = generated_ids.size(1)
        generated_ids = generated_ids.view(batch_size, num_generated, seq_len)
        generated_scores = generated_scores.view(batch_size, num_generated)
        
        # ===== Compute quantile =====
        K = num_generated
        quantiles = generated_scores[:, K-1]  # [B]
        
        # ===== Add ground truth with its score =====
        if self.add_gt:
            with torch.no_grad():
                gt_outputs = self.model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    labels=target_labels,
                    return_dict=True,
                )
                gt_logits = gt_outputs.logits
                
                gt_log_probs = torch.gather(
                    gt_logits.log_softmax(-1),
                    dim=2,
                    index=target_labels.unsqueeze(-1)
                ).squeeze(-1)
                
                gt_mask = (target_labels != self.pad_token_id).float()
                gt_sequence_scores = (gt_log_probs * gt_mask).sum(dim=1)
            
            gt_ids = target_labels.unsqueeze(1)  # [B, 1, L]
            gt_sequence_scores = gt_sequence_scores.unsqueeze(1)  # [B, 1]
            
            # Merge
            all_ids = torch.cat([generated_ids, gt_ids], dim=1)  # [B, num_generated+1, L]
            all_scores = torch.cat([generated_scores, gt_sequence_scores], dim=1)  # [B, num_generated+1]
            num_seqs_per_sample = num_generated + 1
        else:
            all_ids = generated_ids
            all_scores = generated_scores
            num_seqs_per_sample = num_generated
        
        # ===== Flatten =====
        all_ids_flat = all_ids.view(-1, seq_len)  # [B * num_seqs, L]
        all_scores_flat = all_scores.view(-1)  # [B * num_seqs]
        
        # ===== Mask after EOS =====
        is_eos = all_ids_flat == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # ===== Compute advantages =====
        # Get all item IDs
        all_items = []
        for i in range(all_ids_flat.size(0)):
            tokens = all_ids_flat[i].cpu().tolist()
            item = self._tokens_to_item(tokens)
            all_items.append(item if item is not None else -1)
        all_items = torch.tensor(all_items, device=device)
        
        # GT item IDs
        gt_items_list = []
        for i in range(batch_size):
            tokens = target_labels[i].cpu().tolist()
            item = self._tokens_to_item(tokens)
            gt_items_list.append(item if item is not None else -1)
        gt_items_expanded = torch.tensor(
            [gt_items_list[i // num_seqs_per_sample] for i in range(batch_size * num_seqs_per_sample)],
            device=device
        )
        
        # Positive/negative labels
        is_positive = (all_items == gt_items_expanded).float()
        
        # Compute advantage
        all_scores_reshaped = all_scores_flat.view(batch_size, num_seqs_per_sample)
        is_positive_reshaped = is_positive.view(batch_size, num_seqs_per_sample)
        delta = torch.sigmoid((all_scores_reshaped - quantiles.unsqueeze(1)) / self.tau)
        pos_advantage = is_positive_reshaped.float()
        delta_sum = delta.sum(dim=1, keepdim=True)
        neg_advantage = -delta * (delta / (delta_sum + 1e-8)) * (1 - is_positive_reshaped)
        advantages = (pos_advantage + neg_advantage).view(-1)
        
        # Gather
        advantages = gather(advantages)
        all_scores_gathered = gather(all_scores_flat)
        is_positive_gathered = gather(is_positive)
        
        process_slice = slice(
            self.accelerator.process_index * batch_size * num_seqs_per_sample,
            (self.accelerator.process_index + 1) * batch_size * num_seqs_per_sample,
        )
        advantages = advantages[process_slice]
        
        # ===== Reference model log probs =====
        with torch.no_grad():
            encoder_input_ids_expanded = encoder_input_ids.repeat_interleave(num_seqs_per_sample, dim=0)
            encoder_attention_mask_expanded = encoder_attention_mask.repeat_interleave(num_seqs_per_sample, dim=0)
            
            ref_outputs = self.ref_model(
                input_ids=encoder_input_ids_expanded,
                attention_mask=encoder_attention_mask_expanded,
                labels=all_ids_flat,
                return_dict=True,
            )
            ref_logits = ref_outputs.logits
            
            ref_per_token_logps = torch.gather(
                ref_logits.log_softmax(-1),
                dim=2,
                index=all_ids_flat.unsqueeze(-1)
            ).squeeze(-1)
        
        # ===== Log metrics =====
        self._metrics["mean_score"].append(all_scores_gathered.mean().item())
        self._metrics["quantile"].append(quantiles.mean().item())
        self._metrics["advantage_mean"].append(advantages.mean().item())
        self._metrics["advantage_std"].append(advantages.std().item())
        
        if is_positive_gathered.sum() > 0:
            pos_scores = all_scores_gathered[is_positive_gathered.bool()]
            self._metrics["pos_score_mean"].append(pos_scores.mean().item())
        
        if (1 - is_positive_gathered).sum() > 0:
            neg_scores = all_scores_gathered[(1 - is_positive_gathered).bool()]
            self._metrics["neg_score_mean"].append(neg_scores.mean().item())
        
        accuracy = is_positive_gathered.mean().item()
        self._metrics["accuracy"].append(accuracy)
        
        unique_items = len(set(all_items.cpu().tolist()))
        total_items = (all_items != -1).sum().item()
        diversity = unique_items / total_items if total_items > 0 else 0.0
        self._metrics["diversity"].append(diversity)
        
        return {
            "encoder_input_ids": encoder_input_ids_expanded,
            "encoder_attention_mask": encoder_attention_mask_expanded,
            "decoder_input_ids": all_ids_flat,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute RankPO loss.
        
        Loss = Policy Loss + β * KL Divergence
        (Same as GRPO, but advantages are computed differently)
        """
        if return_outputs:
            raise ValueError("RankPOTrainer does not support returning outputs")
        
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
        
        # Compute RankPO loss
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