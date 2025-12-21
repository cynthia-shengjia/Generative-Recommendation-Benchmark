from genrec.quantization.optimizers.base_optimizer import AbstractTokenizerOptimizer
from genrec.quantization.tokenizers.base_tokenizer import AbstractTokenizer
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
from collections import Counter
from k_means_constrained import KMeansConstrained
class LETTERRQVAETrainer:
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

        self.save_best_on = self.config.get('save_best_on', 'collision_rate').lower()
        if self.save_best_on not in ['utilization', 'collision_rate']:
            raise ValueError(f"Invalid 'save_best_on' value: {self.save_best_on}. "
                             f"Must be 'utilization' or 'collision_rate'.")
        
        self.best_metric_value = 0.0 
        self.best_epoch = 0
        logging.info(f"The best model will be saved based on the best value of '{self.save_best_on}'.")

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.tokenizer.to(self.device)
        self.optimizer.move_optimizer_state_to_device(self.device)
        
        self.labels = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": []}

        self.cf_emb_path = self.config.get('cf_emb_path')
        if self.cf_emb_path and os.path.exists(self.cf_emb_path):
            self.cf_embedding = torch.load(self.cf_emb_path).squeeze().detach().numpy()
            logging.info(f"Successfully loaded CF embedding from: {self.cf_emb_path}.")
        else:
            raise FileNotFoundError(f"[ERROR] CF embedding path does not exist: {self.cf_emb_path}")
    def _calculate_codebook_utilization(self, train_dataloader, log_output=True):
        """计算每层码本的利用率"""
        self.tokenizer.eval()
        codebook_usage = [Counter() for _ in range(self.tokenizer.n_codebooks)]
        total_samples = 0
        with torch.no_grad():
            for _, embeddings in train_dataloader:
                embeddings = embeddings.to(self.device)
                indices = self.tokenizer.encode(embeddings)
                for layer_idx in range(self.tokenizer.n_codebooks):
                    layer_indices = indices[:, layer_idx].cpu().numpy()
                    for idx in layer_indices:
                        codebook_usage[layer_idx][idx] += 1
                total_samples += embeddings.size(0)
        
        utilization_rates = [len(usage) / self.tokenizer.codebook_size for usage in codebook_usage]
        if log_output:
            for i, rate in enumerate(utilization_rates):
                logging.info(f"Layer {i + 1}: {len(codebook_usage[i])}/{self.tokenizer.codebook_size} codes used, "
                             f"utilization rate: {rate:.4f}")
        avg_utilization = np.mean(utilization_rates)
        if log_output:
            logging.info(f"Average codebook utilization rate: {avg_utilization:.4f}")
        return utilization_rates, avg_utilization

    def _calculate_collision_rate(self, train_dataloader, log_output=True):
        """计算量化后ID的碰撞率"""
        self.tokenizer.eval()
        indices_set = set()
        total_samples = 0
        with torch.no_grad():
            for _, embeddings in train_dataloader:
                embeddings = embeddings.to(self.device)
                indices = self.tokenizer.encode(embeddings)
                total_samples += embeddings.size(0)
                cpu_indices = indices.cpu().numpy()
                for index_tuple in cpu_indices:
                    code = "-".join(map(str, index_tuple))
                    indices_set.add(code)
        
        if total_samples == 0:
            collision_rate = 0.0
        else:
            collision_rate = (total_samples - len(indices_set)) / total_samples
        
        if log_output:
            logging.info(f"Collision Analysis: {total_samples - len(indices_set)} collisions found for {total_samples} samples.")
            logging.info(f"Total Unique Codes: {len(indices_set)}")
            logging.info(f"Collision rate: {collision_rate:.4f}")
        return collision_rate
    def constrained_km(self, data, n_cluster=10):
        x = data
        size_min = min(len(data) // (n_cluster * 2), 10)
        clf = KMeansConstrained(n_clusters=n_cluster, size_min=size_min, size_max=n_cluster * 6, 
                                max_iter=10, n_init=10, n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels
    def _train_one_epoch(self, train_dataloader, epoch: int):
        self.tokenizer.train()
        total_loss, total_recon_loss, total_commit_loss, total_cf_loss = 0.0, 0.0, 0.0, 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Training]", leave=False)

        # 先对 rq.vq_layers 的 embedding 做聚类，得到 labels
        embs = [layer.embedding.weight.cpu().detach().numpy() for layer in self.tokenizer.rq_vae.rq.vq_layers]
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels

        for batch_idx, data in enumerate(progress_bar):
            emb_idx, embeddings = data[0], data[1]
            embeddings = embeddings.to(self.device)
            self.optimizer.zero_grad()
            tokenizer_output = self.tokenizer(embeddings, self.labels)
            cf_embedding_in_batch = self.cf_embedding[emb_idx]
            cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(self.device)
            loss, reconstruction_loss, commit_loss, cf_loss = self.optimizer.compute_loss(embeddings, cf_embedding_in_batch, tokenizer_output)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()
            total_commit_loss += commit_loss.item()
            total_cf_loss += cf_loss.item()
            if len(progress_bar) % self.log_interval == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'recon_loss': f'{reconstruction_loss.item():.4f}', 'commit_loss': f'{commit_loss.item():.4f}', 'cf_loss': f'{cf_loss.item():.4f}'})
        return total_loss / len(train_dataloader), total_recon_loss / len(train_dataloader), total_commit_loss / len(train_dataloader), total_cf_loss / len(train_dataloader)


    def _save_checkpoint(self, epoch, metric_value=None, is_best=False, utilization_rate=None, collision_rate=None):
        """保存checkpoint"""
        if is_best:
            torch.save(self.tokenizer.state_dict(), self.checkpoint_path)
            logging.info(f"Best model saved to {self.checkpoint_path} ({self.save_best_on}: {metric_value:.4f})")
            return self.checkpoint_path
        else:
            base_path, ext = os.path.splitext(self.checkpoint_path)
            
            # 构造包含指标的文件名
            if utilization_rate is not None and collision_rate is not None:
                checkpoint_filename = f"{base_path}_epoch{epoch+1}_util{utilization_rate:.4f}_coll{collision_rate:.4f}{ext}"
            else:
                checkpoint_filename = f"{base_path}_epoch{epoch+1}{ext}"
                
            torch.save(self.tokenizer.state_dict(), checkpoint_filename)
            logging.info(f"Checkpoint saved to {checkpoint_filename}")
            return checkpoint_filename

    def fit(self, train_dataloader):
        logging.info("Start Training Tokenizer...")
        for epoch in range(self.epochs):
            train_loss, train_recon, train_commit, train_cf = self._train_one_epoch(train_dataloader, epoch)
            logging.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | "
                         f"Train Recon Loss: {train_recon:.4f} | Train Commit Loss: {train_commit:.4f} | Train CF Loss: {train_cf:.4f}")

            if (epoch + 1) % 1000 == 0:
                logging.info(f"\n=== Metrics Analysis at Epoch {epoch+1} ===")
                self._calculate_codebook_utilization(train_dataloader, log_output=True)
                self._calculate_collision_rate(train_dataloader, log_output=True)
                logging.info("=" * 60)

            if (epoch + 1) % self.save_interval == 0:
                # 计算当前周期的指标
                _, avg_utilization = self._calculate_codebook_utilization(train_dataloader, log_output=False)
                collision_rate = self._calculate_collision_rate(train_dataloader, log_output=False)
                
                if self.save_best_on == 'utilization':
                    comparable_metric = avg_utilization
                    original_metric_for_log = avg_utilization
                else: # collision_rate
                    comparable_metric = 1.0 - collision_rate
                    original_metric_for_log = collision_rate

                # 统一的比较逻辑：越大越好
                if comparable_metric >= self.best_metric_value:
                    self.best_metric_value = comparable_metric
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, metric_value=original_metric_for_log, is_best=True)
                    logging.info(f"New best model found at epoch {epoch+1} with {self.save_best_on}: {original_metric_for_log:.4f}")
                
                # 保存包含指标信息的常规 checkpoint
                self._save_checkpoint(epoch, utilization_rate=avg_utilization, collision_rate=collision_rate) 
                        
        logging.info("\n=== Final Metrics Analysis ===")
        _, final_avg_utilization = self._calculate_codebook_utilization(train_dataloader, log_output=True)
        final_collision_rate = self._calculate_collision_rate(train_dataloader, log_output=True)
        logging.info("=" * 60)
        
        self._save_checkpoint(self.epochs - 1, utilization_rate=final_avg_utilization, collision_rate=final_collision_rate)
        
        if self.save_best_on == 'utilization':
            final_comparable_metric = final_avg_utilization
            final_original_metric = final_avg_utilization
        else: # collision_rate
            final_comparable_metric = 1.0 - final_collision_rate
            final_original_metric = final_collision_rate

        if final_comparable_metric > self.best_metric_value:
            self.best_metric_value = final_comparable_metric
            self.best_epoch = self.epochs - 1
            self._save_checkpoint(self.epochs - 1, metric_value=final_original_metric, is_best=True)
            logging.info(f"Final model is the best with {self.save_best_on}: {final_original_metric:.4f}")
        else:
            # 获取用于日志记录的原始最佳值
            best_original_value = self.best_metric_value if self.save_best_on == 'utilization' else 1.0 - self.best_metric_value
            logging.info(f"Best model was at epoch {self.best_epoch+1} with {self.save_best_on}: {best_original_value:.4f}")
        
        best_original_value_final = self.best_metric_value if self.save_best_on == 'utilization' else 1.0 - self.best_metric_value
        logging.info(f"Training complete. Best {self.save_best_on}: {best_original_value_final:.4f} at epoch {self.best_epoch+1}")