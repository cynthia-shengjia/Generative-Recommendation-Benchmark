
from optimizers.tokenizer_optimizers.TokenzierOptimizer import AbstractTokenizerOptimizer
import torch
import torch.nn.functional as F

class RQVAETokenizerOptimizer(AbstractTokenizerOptimizer):

    def __init__(self, config: dict, tokenizer: torch.nn.Module):
        super().__init__(config)
        self.tokenizer = tokenizer
        
        self.commitment_beta = self.config['commitment_beta']
        self.cf_alpha = self.config['cf_alpha']
        learning_rate = self.config['learning_rate']
        
        self.torch_optimizer = torch.optim.Adagrad(self.tokenizer.parameters(), lr=learning_rate)

    def zero_grad(self):
        self.torch_optimizer.zero_grad()
    
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

        total_loss = reconstruction_loss + self.commitment_beta * commit_loss + self.cf_alpha * cf_loss
        
        return total_loss, reconstruction_loss, commit_loss, cf_loss

    def step(self):
        self.torch_optimizer.step()
    
        # 添加一个方法，用于将优化器状态移动到指定设备
    def move_optimizer_state_to_device(self, device):
        for state in self.torch_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)