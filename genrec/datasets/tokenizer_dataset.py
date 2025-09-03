from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import torch
import numpy as np
from typing import Callable, Optional, Dict, List, Any, Tuple, Union


class ItemEmbeddingDataset(Dataset):
    """
    A dataset class for converting item text to embeddings using any Hugging Face Transformers model.
    This class handles item text data and converts it to embeddings for training or inference.
    """
    
    def __init__(
        self,
        data_text_files: str,
        config: Any,
        text_encoder_model: str = "t5-small",
        embedding_extraction_strategy: str = "mean_pooling",
        max_seq_length: int = 512,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the dataset with item text data and a text encoder model.
        
        Args:
            data_text_files: Path to the item text data pickle file
            config: Configuration object
            text_encoder_model: Name of the Hugging Face model to use for text encoding
            embedding_extraction_strategy: Strategy for extracting embeddings from model output
            max_seq_length: Maximum sequence length for tokenization
            device: Device to run the model on (cuda/cpu)
            
        Returns:
            None
        """
        self.config = config
        self.data_text_files = data_text_files
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

        # Read item text information
        self.item_reviews = self._load_item_reviews()
        
        # Get all item IDs
        self.item_ids = list(self.item_reviews.keys())
        
        # Transform item text to item embeddings
        self.item_embeddings = self._transform_semantic_embeddings()
    
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
    
    def _transform_semantic_embeddings(self) -> Dict[int, np.ndarray]:
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
        Get the number of items in the dataset.
        
        Args:
            None
            
        Returns:
            Number of items in the dataset
        """
        return len(self.item_ids)
    
    def __getitem__(self, index: int) -> Dict[str, Union[int, torch.Tensor, str]]:
        """
        Get a single item from the dataset.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
                - item_id: ID of the item
                - embedding: Tensor of the item's embedding
                - text: Original text of the item
        """
        # Get item ID
        item_id = self.item_ids[index]
        
        # Get item embedding and text
        embedding = self.item_embeddings.get(item_id, np.zeros(self.text_model.config.hidden_size))
        text = self.item_reviews.get(item_id, "")
        
        # Convert to tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
        return {
            "item_id": item_id,
            "embedding": embedding_tensor,
            "text": text
        }
    
    def get_embedding_by_id(self, item_id: int) -> Optional[torch.Tensor]:
        """
        Get the embedding for a specific item ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            Tensor containing the item's embedding, or None if not found
        """
        if item_id in self.item_embeddings:
            return torch.tensor(self.item_embeddings[item_id], dtype=torch.float32)
        else:
            raise KeyError(f"Item ID {item_id} not found in embeddings")
    
    def get_text_by_id(self, item_id: int) -> Optional[str]:
        """
        Get the text for a specific item ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            String containing the item's text, or None if not found
        """
        if item_id in self.item_reviews:
            return self.item_reviews.get(item_id, None)
        else:
            raise KeyError(f"Item ID {item_id} not found in item_reviews")


def item_collate_fn(batch: List[Dict[str, Union[int, torch.Tensor, str]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching item embeddings.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary containing:
            - item_ids: Tensor of item IDs
            - embeddings: Tensor of item embeddings
            - texts: List of item texts
    """
    item_ids = torch.tensor([item["item_id"] for item in batch])
    embeddings = torch.stack([item["embedding"] for item in batch])
    texts = [item["text"] for item in batch]
    
    return {
        "item_ids": item_ids,
        "embeddings": embeddings,
        "texts": texts
    }


# Example usage
def create_item_dataloader(
    data_text_files: str,
    config: Any,
    batch_size: int = 32,
    text_encoder_model: str = "t5-small",
    embedding_strategy: str = "mean_pooling"
) -> Tuple[ItemEmbeddingDataset, DataLoader]:
    """
    Create a DataLoader for the item embedding dataset.
    
    Args:
        data_text_files: Path to the item text data pickle file
        config: Configuration object
        batch_size: Batch size for the DataLoader
        text_encoder_model: Name of the Hugging Face model to use for text encoding
        embedding_strategy: Strategy for extracting embeddings from model output
        
    Returns:
        Tuple containing:
            - The item embedding dataset
            - DataLoader for the item embedding dataset
    """
    dataset = ItemEmbeddingDataset(
        data_text_files=data_text_files,
        config=config,
        text_encoder_model=text_encoder_model,
        embedding_extraction_strategy=embedding_strategy
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=item_collate_fn
    )
    
    return dataset, dataloader


