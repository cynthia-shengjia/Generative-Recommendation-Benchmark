
from optimizers.tokenizer_optimizers.TokenzierOptimizer import AbstractTokenizerOptimizer
from genrec.tokenizers.GRTokenizer import AbstractTokenizer
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
from collections import Counter
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
        self.save_interval = self.config.get('save_interval')

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.tokenizer.to(self.device)
        self.optimizer.move_optimizer_state_to_device(self.device)

        # 用于跟踪最佳模型
        self.best_utilization = 0.0
        self.best_epoch = 0

    def _calculate_codebook_utilization(self, train_dataloader, log_output=True):
        """计算每层码本的利用率"""
        self.tokenizer.eval()
        
        # 存储每层的码本使用情况
        codebook_usage = [Counter() for _ in range(self.tokenizer.n_codebooks)]
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (item_ids, embeddings) in enumerate(train_dataloader):
                embeddings = embeddings.to(self.device)
                
                # 获取量化索引
                indices = self.tokenizer.encode(embeddings)  # shape: [batch_size, n_codebooks]
                
                # 统计每层码本的使用情况
                for layer_idx in range(self.tokenizer.n_codebooks):
                    layer_indices = indices[:, layer_idx].cpu().numpy()
                    for idx in layer_indices:
                        codebook_usage[layer_idx][idx] += 1
                
                total_samples += embeddings.size(0)
        
        # 计算利用率
        utilization_rates = []
        for layer_idx, usage in enumerate(codebook_usage):
            used_codes = len(usage)
            total_codes = self.tokenizer.codebook_size
            utilization_rate = used_codes / total_codes
            utilization_rates.append(utilization_rate)
            
            if log_output:
                logging.info(f"Layer {layer_idx + 1}: {used_codes}/{total_codes} codes used, "
                            f"utilization rate: {utilization_rate:.4f}")
        
        avg_utilization = np.mean(utilization_rates)
        if log_output:
            logging.info(f"Average codebook utilization rate: {avg_utilization:.4f}")
        
        return utilization_rates, avg_utilization

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

    def _save_checkpoint(self, epoch, avg_utilization=None, is_best=False):
        """保存checkpoint"""
        if is_best:
            # 最佳模型直接保存到配置的checkpoint_path
            torch.save(self.tokenizer.state_dict(), self.checkpoint_path)
            logging.info(f"Best model saved to {self.checkpoint_path} (utilization: {avg_utilization:.4f})")
            return self.checkpoint_path
        else:
            # 普通checkpoint包含epoch和利用率信息
            base_path, ext = os.path.splitext(self.checkpoint_path)
            checkpoint_filename = f"{base_path}_epoch{epoch+1}{ext}"
            torch.save(self.tokenizer.state_dict(), checkpoint_filename)
            logging.info(f"Checkpoint saved to {checkpoint_filename}")
            return checkpoint_filename

    def fit(self, train_dataloader):
        logging.info("Start Training Tokenizer...")

        for epoch in range(self.epochs):
            train_loss, train_recon, train_commit = self._train_one_epoch(train_dataloader, epoch)
            logging.info(
                f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | "
                f"Train Recon Loss: {train_recon:.4f} | Train Commit Loss: {train_commit:.4f}"
            )
            # 每1000个epoch输出码本利用率
            if (epoch + 1) % 1000 == 0:
                logging.info(f"\n=== Codebook Utilization Analysis at Epoch {epoch+1} ===")
                utilization_rates, avg_utilization = self._calculate_codebook_utilization(train_dataloader, log_output=True)
                logging.info("=" * 60)
            # 每save_interval个epoch保存checkpoint
            if (epoch + 1) % self.save_interval == 0:
                utilization_rates, avg_utilization = self._calculate_codebook_utilization(train_dataloader, log_output=False)
                self._save_checkpoint(epoch, avg_utilization)

                if avg_utilization >= self.best_utilization:
                    self.best_utilization = avg_utilization
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, avg_utilization, is_best=True)
                    logging.info(f"New best model found at epoch {epoch+1} with utilization: {avg_utilization:.4f}")
                    
        # 最终码本利用率
        logging.info("\n=== Final Codebook Utilization Analysis ===")
        final_utilization_rates, final_avg_utilization = self._calculate_codebook_utilization(train_dataloader, log_output=True)
        logging.info("=" * 60)
        
        # 保存最终模型checkpoint
        self._save_checkpoint(self.epochs - 1, final_avg_utilization)
        
        # 如果最终模型比之前的最佳模型更好，也保存为最佳模型
        if final_avg_utilization > self.best_utilization:
            self.best_utilization = final_avg_utilization
            self.best_epoch = self.epochs - 1
            self._save_checkpoint(self.epochs - 1, final_avg_utilization, is_best=True)
            logging.info(f"Final model is the best with utilization: {final_avg_utilization:.4f}")
        else:
            logging.info(f"Best model was at epoch {self.best_epoch+1} with utilization: {self.best_utilization:.4f}")
        
        logging.info(f"Training complete. Best utilization: {self.best_utilization:.4f} at epoch {self.best_epoch+1}")