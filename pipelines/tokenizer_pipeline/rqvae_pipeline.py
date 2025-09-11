import os
import torch
import numpy as np
import pandas as pd
import pickle
import shutil
from torch.utils.data import DataLoader

from genrec.tokenizers.TigerTokenizer import TigerTokenizer
from optimizers.tokenizer_optimizers.TigerOptimizer import RQVAETokenizerOptimizer
from trainers.tokenizer_trainers.TigerTrainer import Trainer
from genrec.datasets.tokenizer_dataset import ItemEmbeddingDataset, item_collate_fn, create_item_dataloader

class DataloaderWrapper:
    def __init__(self, dl):
        self.dl = dl

    def __iter__(self):
        for batch in self.dl:
            yield batch['item_ids'], batch['embeddings']

    def __len__(self):
        return len(self.dl)


class RQVAETrainingPipeline:
    """
    A pipeline that encapsulates the complete training process for an RQ-VAE Tokenizer.
    
    By calling the .run() method, the following steps can be completed in one go:
    1. Data loading and preprocessing
    2. Initialization of the model, optimizer, and trainer
    3. Codebook initialization using K-Means
    4. Model training
    5. Finalization and verification of the Tokenizer
    """
    def __init__(self, config):
        """
        Initializes the pipeline.
        
        Args:
            config (dict): A configuration dictionary containing all model and training parameters.
                           It must include valid data and output paths.
        """
        self.config = config
        
        self.tokenizer = None
        self.optimizer = None
        self.trainer = None
        self.dataset = None
        self.train_dataloader = None

    def _prepare_data(self):
        """Creates the dataset and dataloader."""
        print("\n--- Creating Dataset and Dataloader ---")
        required_paths = ['data_text_files', 'interaction_files','save_path', 'checkpoint_path', 'text_encoder_model']
        for path_key in required_paths:
            if not self.config.get(path_key):
                raise ValueError(f"Configuration error: '{path_key}' must be specified in the config.")

        dataset, dataloader = create_item_dataloader(
            data_text_files=self.config['data_text_files'],
            config=self.config,
            batch_size=self.config['batch_size'],
            text_encoder_model=self.config["text_encoder_model"],
            embedding_strategy=self.config.get("embedding_strategy", "mean_pooling")
        )
        self.dataset = dataset
        self.train_dataloader = DataloaderWrapper(dataloader)
        print("Dataset and Dataloader created successfully.")

        print("\n--- Running a definitive data check ---")
        try:
            _, check_embeddings = next(iter(self.train_dataloader))
            print(f"Batch shape of embeddings from dataloader: {check_embeddings.shape}")
            assert len(check_embeddings.shape) == 2, "Embeddings should be a 2D tensor."
            assert check_embeddings.shape[1] == self.config['sent_emb_dim'], "Embedding dimension mismatch."
            print("--- Data check passed. ---")
        except Exception as e:
            print(f"Data check failed: {e}")
            raise

    def _initialize_components(self):
        """Initializes the model, optimizer, and trainer."""
        print("\n--- Initializing Model, Optimizer, and Trainer ---")
        self.tokenizer = TigerTokenizer(self.config)
        self.optimizer = RQVAETokenizerOptimizer(self.config, self.tokenizer)
        self.trainer = Trainer(self.config, self.tokenizer, self.optimizer)
        print("Initialization complete.")

    def _initialize_codebooks(self):
        """Initializes the RQ-VAE codebooks using K-Means."""
        print("\n--- Initializing RQ-VAE codebooks with K-Means ---")
        all_embeddings = np.vstack([self.dataset.item_embeddings[item_id] for item_id in self.dataset.item_ids])
        print(f"Total embeddings shape for K-Means: {all_embeddings.shape}")
        self.tokenizer.initialize_rqvae(all_embeddings)
        print("Codebook initialization complete.")

    def _train(self):
        """Executes the training loop."""
        print("\n--- Starting Tokenizer Training ---")
        self.trainer.fit(self.train_dataloader)
        print("Training finished.")

    def _finalize_and_verify(self):
        """Finalizes the Tokenizer and verifies its functionality."""
        print("\n--- Finalizing and Testing Tokenizer ---")
        user2item_path = self.config['interaction_files']
        with open(user2item_path, 'rb') as f:
            user2item_data = pickle.load(f)
        user_id_column = user2item_data['UserID']
        all_user_ids = user_id_column.tolist()
        print(f"Extracted {len(all_user_ids)} unique user IDs.")
        item_ids_list = self.dataset.item_ids
        embeddings_array = np.array([self.dataset.item_embeddings[id] for id in item_ids_list])

        self.tokenizer.finalize_tokenization((item_ids_list, embeddings_array), all_user_ids)
        print(f"Tokenizer finalized. Item to token map saved to: {self.config['save_path']}")

    def run(self):
        """
        Executes the complete training pipeline in sequence.
        """
        print(f"--- Starting RQ-VAE Training Pipeline ---")
        print(f"Using device: {self.config.get('device', 'cpu')}")
        
        self._prepare_data()
        self._initialize_components()
        checkpoint_path = self.config.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n--- Checkpoint found at '{checkpoint_path}'. ---")
            print("--- Loading model from checkpoint and skipping training. ---")
            try:
                device = self.config.get('device', 'cpu')
                original_state_dict = torch.load(checkpoint_path, map_location=device)
                new_state_dict = {}
                prefix = 'rq_vae.'
                for key, value in original_state_dict.items():
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                self.tokenizer.rq_vae.load_state_dict(new_state_dict)
                self.tokenizer.rq_vae.to(device)
                print("--- Model loaded successfully. ---")
            except Exception as e:
                print(f"--- Error loading checkpoint: {e} ---")
                print("--- Proceeding with full training pipeline instead. ---")
                self._initialize_codebooks()
                self._train()
                if checkpoint_path and os.path.exists(checkpoint_path):
                    print(f"\n--- Best usage checkpoint found at '{checkpoint_path}'. ---")
                    try:
                        device = self.config.get('device', 'cpu')
                        original_state_dict = torch.load(checkpoint_path, map_location=device)
                        new_state_dict = {}
                        prefix = 'rq_vae.'
                        for key, value in original_state_dict.items():
                            if key.startswith(prefix):
                                new_key = key[len(prefix):]
                                new_state_dict[new_key] = value
                            else:
                                new_state_dict[key] = value
                        self.tokenizer.rq_vae.load_state_dict(new_state_dict)
                        self.tokenizer.rq_vae.to(device)
                        print("--- Best usage model loaded successfully. ---")
                    except Exception as e:
                        print(f"--- Error loading checkpoint: {e} ---")
        else:
            print("\n--- No checkpoint found. Proceeding with full training. ---")
            self._initialize_codebooks()
            self._train()
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"\n--- Best usage checkpoint found at '{checkpoint_path}'. ---")
                try:
                    device = self.config.get('device', 'cpu')
                    original_state_dict = torch.load(checkpoint_path, map_location=device)
                    new_state_dict = {}
                    prefix = 'rq_vae.'
                    for key, value in original_state_dict.items():
                        if key.startswith(prefix):
                            new_key = key[len(prefix):]
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    self.tokenizer.rq_vae.load_state_dict(new_state_dict)
                    self.tokenizer.rq_vae.to(device)
                    print("--- Best usage model loaded successfully. ---")
                except Exception as e:
                    print(f"--- Error loading checkpoint: {e} ---")
        

        # self._initialize_codebooks()
        # self._train()
        self._finalize_and_verify()
        tokenizer_save_path = self.config['tokenizer_path']
        self.tokenizer.save(tokenizer_save_path)
        
        print("\n--- RQ-VAE Training Pipeline Finished Successfully ---")
