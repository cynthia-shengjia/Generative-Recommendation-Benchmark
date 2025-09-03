from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
import numpy as np
from typing import Callable, Optional, Dict, List, Any, Tuple, Union


class TokenizerAmazonReviews2014Dataset(Dataset):
    """
    A dataset class for loading and processing Amazon reviews data with generic text encoders.
    This class handles both interaction data and item text data, converting text to embeddings
    using any Hugging Face Transformers model.
    """
    
    def __init__(
        self,
        data_interaction_files: str,
        data_text_files: str,
        tokenizer: Any,
        config: Any,
        text_encoder_model: str = "t5-small",
        embedding_extraction_strategy: str = "mean_pooling",
        max_seq_length: int = 512,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the dataset with interaction data, text data, and a text encoder model.
        
        Args:
            data_interaction_files: Path to the interaction data pickle file
            data_text_files: Path to the item text data pickle file
            tokenizer: Tokenizer for the main model
            config: Configuration object
            text_encoder_model: Name of the Hugging Face model to use for text encoding
            embedding_extraction_strategy: Strategy for extracting embeddings from model output
            max_seq_length: Maximum sequence length for tokenization
            device: Device to run the model on (cuda/cpu)
            
        Returns:
            None
        """
        self.config = config
        self.data_interaction_files = data_interaction_files
        self.data_text_files = data_text_files
        self.tokenizer = tokenizer
        self.text_encoder_model_name = text_encoder_model
        self.embedding_strategy = embedding_extraction_strategy
        self.max_seq_length = max_seq_length
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load text encoder model and tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_encoder_model_name)
        self.text_model.to(self.device)
        self.text_model.eval()

        # Read user history sequences and item text information
        self.item_reviews = self._load_item_reviews()
        self.user_seqs = self._load_user_seqs()

        # Transform item text to item embeddings
        self.item_embeddings = self._transform_semantic_ids()
    
    def set_embedding_extraction_strategy(self, strategy: str) -> None:
        """
        Set the strategy for extracting embeddings from the model output.
        
        Args:
            strategy: Embedding extraction strategy. Options: 
                     "mean_pooling", "cls_token", "max_pooling", "last_token"
                     
        Returns:
            None
            
        Raises:
            ValueError: If an invalid strategy is provided
        """
        valid_strategies = ["mean_pooling", "cls_token", "max_pooling", "last_token"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Choose from: {valid_strategies}")
        self.embedding_strategy = strategy
    
    def _load_item_reviews(self) -> Dict[int, str]:
        """
        Load item reviews from the text data file.
        
        Args:
            None
            
        Returns:
            Dictionary mapping item IDs to their corresponding text reviews
        """
        item_reviews = defaultdict(str)
        
        with open(self.data_text_files, 'rb') as f:
            item_titles_dataframe = pickle.load(f)
        
        for _, row in item_titles_dataframe.iterrows():
            item_id = int(row['ItemID'])
            item_context_info = row['Title']
            item_reviews[item_id] = item_context_info
        
        return item_reviews
    
    def _load_user_seqs(self) -> Dict[int, List[int]]:
        """
        Load user interaction sequences from the interaction data file.
        
        Args:
            None
            
        Returns:
            Dictionary mapping user IDs to their interaction sequences (list of item IDs)
        """
        user_seqs = defaultdict(list)

        with open(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)
        
        for _, row in user_seqs_dataframe.iterrows():
            user_id = int(row['UserID'])
            item_seq = list(row["ItemID"])
            user_seqs[user_id] = item_seq

        return user_seqs
    
    def _extract_embeddings(
        self, 
        model_outputs: Any, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embeddings from model outputs based on the selected strategy.
        
        Args:
            model_outputs: Output from the text encoder model
            attention_mask: Attention mask indicating which tokens are padding
            
        Returns:
            Tensor containing the extracted embeddings
            
        Raises:
            ValueError: If an unknown embedding strategy is specified
        """
        last_hidden_state = model_outputs.last_hidden_state
        
        if self.embedding_strategy == "mean_pooling":
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.embedding_strategy == "cls_token":
            # Use the [CLS] token (works for BERT-like models)
            return last_hidden_state[:, 0, :]
        
        elif self.embedding_strategy == "max_pooling":
            # Max pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            # Set padding tokens to large negative value so they don't affect max
            last_hidden_state[input_mask_expanded == 0] = -1e9
            return torch.max(last_hidden_state, 1)[0]
        
        elif self.embedding_strategy == "last_token":
            # Use the last token (before padding)
            sequence_lengths = torch.sum(attention_mask, dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[torch.arange(batch_size), sequence_lengths]
        
        else:
            raise ValueError(f"Unknown embedding strategy: {self.embedding_strategy}")
    
    def _transform_semantic_ids(self) -> Dict[int, np.ndarray]:
        """
        Transform item texts to embeddings using the specified model.
        
        Args:
            None
            
        Returns:
            Dictionary mapping item IDs to their corresponding embeddings
        """
        item_embeddings = {}
        
        # Batch processing for efficiency
        batch_size = 32  # Adjust based on GPU memory
        item_ids = list(self.item_reviews.keys())
        item_texts = [self.item_reviews[item_id] for item_id in item_ids]
        
        for i in range(0, len(item_texts), batch_size):
            batch_texts = item_texts[i:i+batch_size]
            batch_ids = item_ids[i:i+batch_size]
            
            # Tokenize the batch
            inputs = self.text_tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_seq_length
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            
            # Extract embeddings based on the selected strategy
            embeddings = self._extract_embeddings(outputs, inputs.attention_mask)
            
            # Store embeddings
            for j, item_id in enumerate(batch_ids):
                item_embeddings[item_id] = embeddings[j].cpu().numpy()
        
        return item_embeddings
    
    def __len__(self) -> int:
        """
        Get the number of users in the dataset.
        
        Args:
            None
            
        Returns:
            Number of users in the dataset
        """
        return len(self.user_seqs)
    
    def __getitem__(self, index: int) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - user_id: ID of the user
                - sequence_embeddings: Tensor of item embeddings for the user's sequence
                - sequence_length: Length of the sequence
        """
        # Get user ID and sequence
        user_ids = list(self.user_seqs.keys())
        user_id = user_ids[index]
        item_seq = self.user_seqs[user_id]
        
        # Convert each item ID in the sequence to its embedding
        seq_embeddings = []
        for item_id in item_seq:
            if item_id in self.item_embeddings:
                seq_embeddings.append(self.item_embeddings[item_id])
            else:
                # Handle unknown items - use zero vector or special token
                embedding_size = self.text_model.config.hidden_size
                seq_embeddings.append(np.zeros(embedding_size))
        
        # Convert to tensor
        seq_embeddings = torch.tensor(np.array(seq_embeddings), dtype=torch.float32)
        
        return {
            "user_id": user_id,
            "sequence_embeddings": seq_embeddings,
            "sequence_length": len(item_seq)
        }


def collate_fn(batch: List[Dict[str, Union[int, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary containing:
            - padded_sequences: Padded sequence embeddings
            - lengths: Original lengths of the sequences
            - user_ids: User IDs for the batch
    """
    # Sort by sequence length (for pack_padded_sequence)
    batch.sort(key=lambda x: x["sequence_length"], reverse=True)
    
    sequences = [item["sequence_embeddings"] for item in batch]
    user_ids = [item["user_id"] for item in batch]
    lengths = [item["sequence_length"] for item in batch]
    
    # Pad sequences
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    return {
        "padded_sequences": padded_sequences,
        "lengths": lengths,
        "user_ids": user_ids
    }


# Example usage
def create_dataloader(
    data_interaction_files: str,
    data_text_files: str,
    tokenizer: Any,
    config: Any,
    batch_size: int = 32,
    text_encoder_model: str = "t5-small",
    embedding_strategy: str = "mean_pooling"
) -> DataLoader:
    """
    Create a DataLoader for the Amazon Reviews dataset.
    
    Args:
        data_interaction_files: Path to the interaction data pickle file
        data_text_files: Path to the item text data pickle file
        tokenizer: Tokenizer for the main model
        config: Configuration object
        batch_size: Batch size for the DataLoader
        text_encoder_model: Name of the Hugging Face model to use for text encoding
        embedding_strategy: Strategy for extracting embeddings from model output
        
    Returns:
        DataLoader for the Amazon Reviews dataset
    """
    dataset = TokenizerAmazonReviews2014Dataset(
        data_interaction_files=data_interaction_files,
        data_text_files=data_text_files,
        tokenizer=tokenizer,
        config=config,
        text_encoder_model=text_encoder_model,
        embedding_extraction_strategy=embedding_strategy
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )