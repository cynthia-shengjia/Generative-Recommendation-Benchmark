
import torch
import torch.nn.functional as F
from genrec.quantization.optimizers.rqvae_optimizer import RQVAETokenizerOptimizer



class LETTERRQVAETokenizerOptimizer(RQVAETokenizerOptimizer):

    def __init__(self, config: dict, tokenizer: torch.nn.Module):
        super().__init__(config, tokenizer)
        self.cf_alpha = self.config['cf_alpha']

    def CF_loss(self, dense_quantized_rep: torch.Tensor, cf_rep: torch.Tensor):
        batch_size = dense_quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=dense_quantized_rep.device)
        similarities = torch.matmul(dense_quantized_rep, cf_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss
    def compute_loss(self, original_embeddings: torch.Tensor, cf_embeddings: torch.Tensor, tokenizer_output: tuple):
        quantized_embeddings, _, commit_loss, dense_quantized_embeddings = tokenizer_output
        
        # Compute recon loss
        reconstruction_loss = F.mse_loss(quantized_embeddings, original_embeddings)

        # Compute CF loss
        cf_loss = self.CF_loss(dense_quantized_embeddings, cf_embeddings)

        total_loss = reconstruction_loss + self.quant_loss_weight * commit_loss + self.cf_alpha * cf_loss
        
        return total_loss, reconstruction_loss, commit_loss, cf_loss
