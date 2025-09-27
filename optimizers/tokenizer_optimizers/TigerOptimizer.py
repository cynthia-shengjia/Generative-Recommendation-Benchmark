
from optimizers.tokenizer_optimizers.TokenzierOptimizer import AbstractTokenizerOptimizer
import torch
import torch.nn.functional as F

class RQVAETokenizerOptimizer(AbstractTokenizerOptimizer):

    def __init__(self, config: dict, tokenizer: torch.nn.Module):
        super().__init__(config)
        self.tokenizer = tokenizer
        
        self.quant_loss_weight = self.config['quant_loss_weight']
        learning_rate = self.config['learning_rate']
        
        self.torch_optimizer = torch.optim.Adagrad(self.tokenizer.parameters(), lr=learning_rate)

    def zero_grad(self):
        self.torch_optimizer.zero_grad()
    
    def compute_loss(self, original_embeddings: torch.Tensor, tokenizer_output: tuple):
        quantized_embeddings, _, commit_loss = tokenizer_output
        
        reconstruction_loss = F.mse_loss(quantized_embeddings, original_embeddings)
        total_loss = reconstruction_loss + self.quant_loss_weight * commit_loss
        
        return total_loss, reconstruction_loss, commit_loss

    def step(self):
        self.torch_optimizer.step()
    
        # 添加一个方法，用于将优化器状态移动到指定设备
    def move_optimizer_state_to_device(self, device):
        for state in self.torch_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)