from transformers import AutoModel, AutoTokenizer, AutoConfig, T5EncoderModel
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pickle
import torch
import numpy as np
from typing import Callable, Optional, Dict, List, Any, Tuple, Union
from sentence_transformers import SentenceTransformer

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
        if 'sentence' in self.text_encoder_model_name.lower():
            self.text_model = SentenceTransformer(self.text_encoder_model_name)
            self.model_config = self.text_model[0].auto_model.config
        elif 't5' in self.text_encoder_model_name.lower():
            self.text_model = T5EncoderModel.from_pretrained(self.text_encoder_model_name)
            self.model_config = self.text_model.config
        else:
            model_config = AutoConfig.from_pretrained(self.text_encoder_model_name)
            if model_config.is_encoder_decoder:
                # Load the full model and extract the encoder
                full_model = AutoModel.from_pretrained(self.text_encoder_model_name)
                self.text_model = full_model.encoder
                self.model_config = full_model.config
            else:
                # Load encoder-only models directly
                self.text_model = AutoModel.from_pretrained(self.text_encoder_model_name)
                self.model_config = self.text_model.config
        # Load model config first to check architecture
        # model_config = AutoConfig.from_pretrained(self.text_encoder_model_name)
        # self.is_encoder_decoder = model_config.is_encoder_decoder
        
        # Load text encoder model and tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_model_name)
        
        # For encoder-decoder models, we'll only use the encoder part
        # if self.is_encoder_decoder:
        #     # Get the encoder part of the model
        #     full_model = AutoModel.from_pretrained(self.text_encoder_model_name)
        #     self.text_model = full_model.encoder
        #     # Set model config for embedding dimension
        #     self.model_config = full_model.config
        # else:
        #     self.text_model = AutoModel.from_pretrained(self.text_encoder_model_name)
        #     self.model_config = self.text_model.config
            
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
    def _load_item_reviews(self) -> Dict[int, Tuple[str, str]]:
        """
        Load item reviews from the text data file, splitting into basic info and description.
        Handles empty string values gracefully.
        
        Args:
            None
            
        Returns:
            Dictionary mapping item IDs to tuple of (basic_info, description)
        """
        item_reviews = defaultdict(lambda: ("", ""))
        
        with open(self.data_text_files, 'rb') as f:
            item_titles_dataframe = pickle.load(f)
        
        for _, row in item_titles_dataframe.iterrows():
            try:
                item_id = int(row['ItemID'])
                
                def safe_get_field(field_name: str, default: str = "Unknown") -> str:
                    value = row.get(field_name, "")
                    if isinstance(value, list):
                        return ", ".join(value) if value else default
                    if value == "" or str(value).lower() == "nan":
                        return default
                    return str(value).strip()
                
                # 获取各个字段
                title = safe_get_field('Title', 'Unknown')
                categories = safe_get_field('Categories', 'Unknown')
                brand = safe_get_field('Brand', 'Unknown')
                description = safe_get_field('Description', '') 
                year = safe_get_field('Year', 'Unknown')
                Genres = safe_get_field('Genres', 'Unknown')
                basic_info_parts = []
                
                basic_info_parts.append(f"Atomic Item ID: {item_id}")
                
                # 只有非Unknown的字段才添加
                if title != 'Unknown':
                    basic_info_parts.append(f"Title: {title}")
                
                if categories != 'Unknown':
                    basic_info_parts.append(f"Categories: {categories}")
                
                if brand != 'Unknown':
                    basic_info_parts.append(f"Brand: {brand}")
                if year != 'Unknown':
                    basic_info_parts.append(f"Year: {year}")
                if Genres != 'Unknown':
                    basic_info_parts.append(f"Genres: {Genres}")
                
                basic_info = ", ".join(basic_info_parts)
                
                item_reviews[item_id] = (basic_info, description)
                
            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Error processing row: {e}")
                continue
        
        print(f"Successfully loaded {len(item_reviews)} items")
        return item_reviews



    # def _load_item_reviews(self) -> Dict[int, Tuple[str, str]]:
    #     """
    #     Load item reviews from the text data file, splitting into basic info and description.
        
    #     Args:
    #         None
            
    #     Returns:
    #         Dictionary mapping item IDs to tuple of (basic_info, description)
    #     """
    #     item_reviews = defaultdict(lambda: ("", ""))
        
    #     with open(self.data_text_files, 'rb') as f:
    #         item_titles_dataframe = pickle.load(f)
        
    #     for _, row in item_titles_dataframe.iterrows():
    #         item_id = int(row['ItemID'])
    #         # 基本信息部分（不包含描述）
    #         basic_info = f"Atomic Item ID: {row['ItemID']}, Title: {row['Title']}, Categories: {row['Categories']}, Brand: {row['Brand']}"
    #         # 描述部分
    #         description = str(row['Description']) if row['Description'] else ""
            
    #         item_reviews[item_id] = (basic_info, description)
        
    #     return item_reviews
    
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
        # Handle different output formats
        if hasattr(model_outputs, 'last_hidden_state'):
            last_hidden_state = model_outputs.last_hidden_state
        else:
            # For some models, the output might be a tuple
            last_hidden_state = model_outputs[0]
        
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
        Process basic info and description separately, then average them.
        Only process descriptions that are not empty.
        
        Args:
            None
            
        Returns:
            Dictionary mapping item IDs to their corresponding averaged embeddings
        """
        item_embeddings = {}
        
        # Batch processing for efficiency
        batch_size = 32  # Adjust based on GPU memory
        item_ids = list(self.item_reviews.keys())
        
        # 分别提取基本信息和描述
        basic_infos = [self.item_reviews[item_id][0] for item_id in item_ids]
        descriptions = [self.item_reviews[item_id][1] for item_id in item_ids]
        
        print("Processing basic info embeddings...")
        basic_embeddings = self._process_text_batch(basic_infos, item_ids, batch_size)
        
        # print("Processing description embeddings...")
        # # 只处理非空描述
        
        # desc_embeddings = self._process_text_batch(descriptions, item_ids, batch_size)
        desc_embeddings = None
        # 合并嵌入
        for item_id in item_ids:
            basic_emb = basic_embeddings.get(item_id)
            # desc_emb = desc_embeddings.get(item_id)
            
            if basic_emb is None:
                basic_emb = np.zeros(self.model_config.hidden_size)
                
            if False:
                item_embeddings[item_id] = (basic_emb + desc_emb) / 2.0
            else:
                item_embeddings[item_id] = basic_emb
        
        # num_with_desc = len([item_id for item_id in item_ids if desc_embeddings.get(item_id) is not None])
        print(f"Final embeddings generated for {len(item_embeddings)} items")
        # print(f"Items with descriptions: {num_with_desc}")
        # print(f"Items without descriptions: {len(item_embeddings) - num_with_desc}")
        
        return item_embeddings

    # def _process_text_batch(self, texts: List[str], item_ids: List[int], batch_size: int) -> Dict[int, np.ndarray]:
    #     """
    #     Process a batch of texts and return embeddings.
    #     Only processes non-empty texts and returns None for empty texts.
        
    #     Args:
    #         texts: List of texts to process
    #         item_ids: Corresponding item IDs
    #         batch_size: Batch size for processing
            
    #     Returns:
    #         Dictionary mapping item IDs to embeddings (only for non-empty texts)
    #     """
    #     embeddings_dict = {}
        
    #     # 预先过滤空文本，保留原始索引
    #     valid_pairs = []
    #     for i, (text, item_id) in enumerate(zip(texts, item_ids)):
    #         if text and text.strip():  # 只处理非空文本
    #             valid_pairs.append((text, item_id))
        
    #     if not valid_pairs:
    #         print("Warning: No valid texts to process")
    #         return embeddings_dict  # 返回空字典
        
    #     valid_texts, valid_ids = zip(*valid_pairs)
    #     valid_texts = list(valid_texts)
    #     valid_ids = list(valid_ids)
        
    #     print(f"Processing {len(valid_texts)} valid texts out of {len(texts)} total texts...")
        
    #     for i in range(0, len(valid_texts), batch_size):
    #         batch_texts = valid_texts[i:i+batch_size]
    #         batch_ids = valid_ids[i:i+batch_size]
            
    #         # Tokenize the batch
    #         inputs = self.text_tokenizer(
    #             batch_texts, 
    #             return_tensors="pt", 
    #             padding=True, 
    #             truncation=True, 
    #             max_length=self.max_seq_length
    #         ).to(self.device)
            
    #         # Get model outputs
    #         with torch.no_grad():
    #             outputs = self.text_model(**inputs)
            
    #         # Extract embeddings based on the selected strategy
    #         embeddings = self._extract_embeddings(outputs, inputs.attention_mask)
            
    #         # Store embeddings (只存储有效文本的embedding)
    #         for j, item_id in enumerate(batch_ids):
    #             embeddings_dict[item_id] = embeddings[j].cpu().numpy()
        
    #     return embeddings_dict
    def _process_text_batch(self, texts: List[str], item_ids: List[int], batch_size: int) -> Dict[int, np.ndarray]:
        valid_pairs = [(t, i) for t, i in zip(texts, item_ids) if t and t.strip()]
        if not valid_pairs: return {}
        
        valid_texts, valid_ids = zip(*valid_pairs)
        
        # 直接调用 encode，它会自动处理 batching 和 pooling
        embeddings = self.text_model.encode(
            list(valid_texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return {item_id: emb for item_id, emb in zip(valid_ids, embeddings)}
    def __len__(self) -> int:
        """
        Get the number of items in the dataset.
        
        Args:
            None
            
        Returns:
            Number of items in the dataset
        """
        return len(self.item_ids)
    
    # def __getitem__(self, index: int) -> Dict[str, Union[int, torch.Tensor, str]]:
    #     """
    #     Get a single item from the dataset.
        
    #     Args:
    #         index: Index of the item to retrieve
            
    #     Returns:
    #         Dictionary containing:
    #             - item_id: ID of the item
    #             - embedding: Tensor of the item's embedding
    #             - text: Original text of the item
    #     """
    #     # Get item ID
    #     item_id = self.item_ids[index]
        
    #     # Get item embedding and text
    #     embedding = self.item_embeddings.get(item_id, np.zeros(self.model_config.hidden_size))
    #     text = self.item_reviews.get(item_id, "")
        
    #     # Convert to tensor
    #     embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
    #     return {
    #         "item_id": item_id,
    #         "embedding": embedding_tensor,
    #         "text": text
    #     }
    def __getitem__(self, index: int) -> Dict[str, Union[int, torch.Tensor, str]]:
        """
        Get a single item from the dataset.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
                - item_id: ID of the item
                - embedding: Tensor of the item's averaged embedding
                - basic_info: Basic information text
                - description: Description text
                - text: Combined text (for backward compatibility)
        """
        # Get item ID
        item_id = self.item_ids[index]
        
        # Get item embedding and texts
        embedding = self.item_embeddings.get(item_id, np.zeros(self.model_config.hidden_size))
        basic_info, description = self.item_reviews.get(item_id, ("", ""))
        
        # Combined text for backward compatibility
        combined_text = f"{basic_info} Description: {description}" if description else basic_info
        
        # Convert to tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
        return {
            "item_id": item_id,
            "embedding": embedding_tensor,
            "basic_info": basic_info,
            "description": description,
            "text": combined_text
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
    
    def get_text_by_id(self, item_id: int) -> Optional[Tuple[str, str]]:
        """
        Get the text for a specific item ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            Tuple containing (basic_info, description), or None if not found
        """
        if item_id in self.item_reviews:
            return self.item_reviews.get(item_id, ("", ""))
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