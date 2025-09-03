
from optimizers.tokenizer_optimizers.TokenzierOptimizer import AbstractTokenizerOptimizer
from genrec.tokenizers.GRTokenizer import AbstractTokenizer
import torch
import logging
import os
from tqdm import tqdm

class Trainer:
    def __init__(
        self, 
        config: dict,  
        tokenizer: AbstractTokenizer,
        optimizer: AbstractTokenizerOptimizer 
    ):
        self.config   = config
        self.tokenizer  = tokenizer  
        self.optimizer = optimizer

        self.epochs = self.config.get('epochs')
        self.device = torch.device(self.config.get('device'))
        self.log_interval = self.config.get('log_interval')
        self.checkpoint_path = self.config.get('checkpoint_path')
        
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.tokenizer.to(self.device)

    def _train_one_epoch(self, train_dataloader, epoch: int):
        self.tokenizer.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_commit_loss = 0.0

        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{self.epochs} [Training]",
            leave=False
        )

        for batch_idx, (item_ids, embeddings) in enumerate(progress_bar):
            embeddings = embeddings.to(self.device)
            
            self.optimizer.zero_grad()
            tokenizer_output = self.tokenizer(embeddings)

            loss, reconstruction_loss, commit_loss = self.optimizer.compute_loss(embeddings, tokenizer_output)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()
            total_commit_loss += commit_loss.item()
            
            if batch_idx % self.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon_loss': f'{reconstruction_loss.item():.4f}',
                    'commit_loss': f'{commit_loss.item():.4f}'
                })
                
        avg_loss = total_loss / len(train_dataloader)
        avg_recon_loss = total_recon_loss / len(train_dataloader)
        avg_commit_loss = total_commit_loss / len(train_dataloader)
        
        return avg_loss, avg_recon_loss, avg_commit_loss

    def fit(self, train_dataloader):
        logging.info("Start Training Tokenizer...")

        for epoch in range(self.epochs):
            train_loss, train_recon, train_commit = self._train_one_epoch(train_dataloader, epoch)
            logging.info(
                f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | "
                f"Train Recon Loss: {train_recon:.4f} | Train Commit Loss: {train_commit:.4f}"
            )
        
        torch.save(self.tokenizer.state_dict(), self.checkpoint_path)
        logging.info(f"Training complete. Final model saved to {self.checkpoint_path}")