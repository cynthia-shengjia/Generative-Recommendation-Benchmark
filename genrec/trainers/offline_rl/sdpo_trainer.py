# genrec/trainers/offline_rl/sdpo_trainer.py

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from .base_trainer import BaseOfflineRLTrainer

class SDPOTrainer(BaseOfflineRLTrainer):
    """
    S-DPO (Simplified Direct Preference Optimization) Trainer.
    
    S-DPO extends DPO to handle multiple rejected samples simultaneously.
    """
    
    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Optimized Encoder-Decoder forward pass: Encoder computed only once.
        
        Returns:
            chosen_logps: [B]
            rejected_logps: [B, N]
            chosen_logits: [B, L, V]
            rejected_logits: [B, N, L, V]
        """
        batch_size = batch["input_ids"].shape[0]
        num_rejected = batch["rejected_labels"].shape[1]
        
        # ===== 1. Encoder (computed once) =====
        encoder_outputs = model.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        
        # ===== 2. Decoder for Chosen =====
        chosen_outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=batch["attention_mask"],
            labels=batch["chosen_labels"],
        )
        chosen_logits = chosen_outputs.logits.to(torch.float32)  # [B, L, V]
        
        chosen_logps = self._get_batch_logps(
            chosen_logits,
            batch["chosen_labels"],
            average_log_prob=False,
        )  # [B]
        
        # ===== 3. Decoder for Rejected (reuse Encoder outputs) =====
        rejected_labels_flat = batch["rejected_labels"].view(batch_size * num_rejected, -1)
        
        # Repeat encoder hidden states N times
        encoder_hidden_states_repeated = encoder_outputs.last_hidden_state.repeat_interleave(
            num_rejected, dim=0
        )
        
        # Repeat attention_mask N times
        attention_mask_repeated = batch["attention_mask"].repeat_interleave(num_rejected, dim=0)
        
        # Create new encoder_outputs object
        repeated_encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states_repeated,
        )
        
        # Decoder forward pass
        rejected_outputs = model(
            encoder_outputs=repeated_encoder_outputs,
            attention_mask=attention_mask_repeated,
            labels=rejected_labels_flat,
        )
        rejected_logits_flat = rejected_outputs.logits.to(torch.float32)  # [B*N, L, V]
        
        rejected_logps_flat = self._get_batch_logps(
            rejected_logits_flat,
            rejected_labels_flat,
            average_log_prob=False,
        )  # [B*N]
        
        # Reshape: [B*N] -> [B, N]
        rejected_logps = rejected_logps_flat.view(batch_size, num_rejected)
        
        # Reshape logits: [B*N, L, V] -> [B, N, L, V]
        rejected_logits = rejected_logits_flat.view(
            batch_size, num_rejected, rejected_logits_flat.shape[1], rejected_logits_flat.shape[2]
        )
        
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits
    
    def compute_rl_loss(
        self,
        policy_outputs: Tuple,
        reference_outputs: Tuple,
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Compute S-DPO loss.
        
        Args:
            policy_outputs: (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
            reference_outputs: (chosen_logps, rejected_logps, _, _)
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits = policy_outputs
        reference_chosen_logps, reference_rejected_logps, _, _ = reference_outputs
        
        # Compute log-ratios
        chosen_logratios = policy_chosen_logps - reference_chosen_logps  # [B]
        rejected_logratios = policy_rejected_logps - reference_rejected_logps  # [B, N]
        
        # S-DPO core: softmax over all rejected
        chosen_logratios_expanded = chosen_logratios.unsqueeze(1)  # [B, 1]
        logit_diff = self.beta * (rejected_logratios - chosen_logratios_expanded)  # [B, N]
        log_sum_exp = torch.logsumexp(logit_diff, dim=1)  # [B]
        losses = -F.logsigmoid(-log_sum_exp)  # [B]
        
        # Compute rewards (for logging)
        chosen_rewards = self.beta * chosen_logratios.detach()  # [B]
        rejected_rewards = self.beta * rejected_logratios.detach()  # [B, N]
        
        # Compute accuracy: chosen better than all rejected
        reward_comparisons = chosen_rewards.unsqueeze(1) > rejected_rewards  # [B, N]
        reward_accuracies = reward_comparisons.all(dim=1).float()  # [B]
        
        # Metrics
        metrics = {}
        metrics["rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        metrics["rewards/rejected_mean"] = rejected_rewards.cpu().numpy().mean()
        
        num_rejected = rejected_rewards.shape[1]
        for i in range(num_rejected):
            metrics[f"rewards/rejected{i+1}"] = rejected_rewards[:, i].cpu().numpy().mean()
        
        metrics["rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        
        margins = chosen_rewards.unsqueeze(1) - rejected_rewards  # [B, N]
        metrics["rewards/margins_mean"] = margins.cpu().numpy().mean()
        for i in range(num_rejected):
            metrics[f"rewards/margins_rejected{i+1}"] = margins[:, i].cpu().numpy().mean()
        
        metrics["logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        metrics["logps/rejected_mean"] = policy_rejected_logps.detach().cpu().numpy().mean()
        
        metrics["logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()
        metrics["logits/rejected_mean"] = policy_rejected_logits.detach().cpu().numpy().mean()
        
        return losses.mean(), metrics