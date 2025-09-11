import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import abc
import torch.nn.functional as F
import torch
import math
from typing import Dict, Tuple
class HSTUSeqModelTrainingDataset(Dataset):
    def __init__(
        self,
        data_interaction_files: str,
        data_text_files: str,
        tokenizer: Any, # AbstractTokenizer,
        config: dict,
        mode: str = 'train',  # 'train', 'valid', or 'test'
        device: Optional[str] = None
    ) -> None:
        self.config = config
        self.data_interaction_files = data_interaction_files
        self.data_text_files = data_text_files
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device if device else torch.device('cpu')
        
        assert 'max_seq_len' in config, "config must contain 'max_seq_len'"
        
        # 加载数据
        self.item_reviews = self._load_item_reviews()
        # CHANGED: _load_user_seqs 现在会同时加载物品和时间戳
        self.user_seqs = self._load_user_seqs()
        self.user_ids = list(self.user_seqs.keys())
        
        self.tokens_per_item = self._get_tokens_per_item()
        self.max_token_len = (self.tokens_per_item + 1) * self.config['max_seq_len'] + 1
        
        # CHANGED: _create_samples 现在会同时处理物品和时间戳
        self.samples = self._create_samples()

    def _load_item_reviews(self) -> Dict[int, str]:
        # 保持不变
        item_reviews = defaultdict(str)
        with open(self.data_text_files, 'rb') as f:
            item_titles_dataframe = pickle.load(f)
        for _, row in item_titles_dataframe.iterrows():
            item_id = int(row['ItemID'])
            item_context_info = row['Title']
            item_reviews[item_id] = item_context_info
        return item_reviews

    # CHANGED: 此函数现在加载并返回物品和时间戳
    def _load_user_seqs(self) -> Dict[int, Dict[str, list]]:
        """
        加载用户序列数据，同时包含物品ID和时间戳。
        
        **重要假设**: 输入的 pickle 文件是一个 DataFrame，
        其中包含 'UserID', 'ItemID' (list), 和 'Timestamp' (list) 列。
        """
        user_seqs_data = defaultdict(dict)
        with open(self.data_interaction_files, 'rb') as f:
            user_seqs_dataframe = pickle.load(f)

        # 确保时间戳列存在
        if 'Timestamp' not in user_seqs_dataframe.columns:
            raise ValueError("Interaction data file must contain a 'Timestamp' column.")

        for _, row in user_seqs_dataframe.iterrows():
            user_id = int(row['UserID'])
            item_seq = list(row["ItemID"])
            timestamp_seq = list(row["Timestamp"]) # NEW: 加载时间戳序列

            if len(item_seq) != len(timestamp_seq):
                print(f"Warning: User {user_id} has mismatched item and timestamp sequence lengths. Skipping.")
                continue

            user_seqs_data[user_id] = {
                'items': item_seq,
                'timestamps': timestamp_seq
            }
        return user_seqs_data
    
    def _get_tokens_per_item(self) -> int:
        # 保持不变
        if not hasattr(self.tokenizer, 'item2tokens') or not self.tokenizer.item2tokens:
            return 1
        first_item = next(iter(self.tokenizer.item2tokens.keys()))
        return len(self.tokenizer.item2tokens[first_item])

    # CHANGED: 此函数现在同步处理物品和时间戳序列
    def _create_samples(self) -> List[Dict[str, Any]]:
        """创建样本，返回历史物品/时间戳序列和目标物品"""
        samples = []
        max_item_seq_len = self.config['max_seq_len']
        
        for user_id, seq_data in self.user_seqs.items():
            item_seq = seq_data['items']
            timestamp_seq = seq_data['timestamps'] # NEW: 获取时间戳序列

            if self.mode == 'train':
                # 训练集: item_seq[:-2], timestamp_seq[:-2]
                train_item_seq = item_seq[:-2]
                train_timestamp_seq = timestamp_seq[:-2]
                for i in range(1, len(train_item_seq)):
                    history = train_item_seq[:i]
                    history_ts = train_timestamp_seq[:i] # NEW: 同步切分时间戳
                    target = train_item_seq[i]
                    
                    if len(history) > max_item_seq_len:
                        history = history[-max_item_seq_len:]
                        history_ts = history_ts[-max_item_seq_len:] # NEW: 同步截断时间戳

                    samples.append({
                        'user_id': user_id,
                        'history_items': history,
                        'history_timestamps': history_ts, # NEW: 保存历史时间戳
                        'target_item': target
                    })

            elif self.mode == 'valid':
                if len(item_seq) < 3: continue
                history = item_seq[:-2]
                history_ts = timestamp_seq[:-2] # NEW
                target = item_seq[-2]
                if len(history) > max_item_seq_len:
                    history = history[-max_item_seq_len:]
                    history_ts = history_ts[-max_item_seq_len:] # NEW
                samples.append({
                    'user_id': user_id,
                    'history_items': history,
                    'history_timestamps': history_ts, # NEW
                    'target_item': target
                })

            elif self.mode == 'test':
                if len(item_seq) < 2: continue
                history = item_seq[:-1]
                history_ts = timestamp_seq[:-1] # NEW
                target = item_seq[-1]
                if len(history) > max_item_seq_len:
                    history = history[-max_item_seq_len:]
                    history_ts = history_ts[-max_item_seq_len:] # NEW
                samples.append({
                    'user_id': user_id,
                    'history_items': history,
                    'history_timestamps': history_ts, # NEW
                    'target_item': target
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    # CHANGED: 此函数现在返回的数据中包含了 source_timestamps
    def __getitem__(self, index: int) -> Dict[str, Union[int, List[int]]]:
        sample = self.samples[index]
        history_items = sample['history_items']
        history_timestamps = sample['history_timestamps'] # NEW: 获取历史时间戳
        target_item = sample['target_item']
        user_id = sample['user_id']
        
        source_tokens = []
        source_timestamps = [] # NEW: 用于存储与 source_tokens 对齐的时间戳

        # 将历史物品转换为token序列，并同步处理时间戳
        for item, timestamp in zip(history_items, history_timestamps):
            num_tokens_for_item = self.tokens_per_item
            if item in self.tokenizer.item2tokens:
                tokens = self.tokenizer.item2tokens[item]
                source_tokens.extend(tokens)
                num_tokens_for_item = len(tokens)
            else:
                source_tokens.extend([0] * self.tokens_per_item)

            # NEW: 为该物品的每个 token 复制其时间戳
            source_timestamps.extend([timestamp] * num_tokens_for_item)
        
        # 将目标物品转换为token序列
        if target_item in self.tokenizer.item2tokens:
            target_tokens = self.tokenizer.item2tokens[target_item]
        else:
            target_tokens = [0] * self.tokens_per_item
        
        return {
            'user_token': self.tokenizer.get_user_token(user_id),
            'source_tokens': source_tokens,
            'source_timestamps': source_timestamps, # NEW: 返回给 DataCollator
            'target_tokens': target_tokens,
            "target_id": target_item
        }
    

@dataclass
class HSTUDataCollator:
    """
    Data Collator for the HSTU Encoder-Decoder model.

    This collator performs the following operations:
    1.  Processes the source sequence (user history) by prepending a user-specific token,
        then truncating or LEFT-padding to `max_seq_len`.
    2.  Processes the corresponding timestamps for the source sequence, applying the exact
        same truncation and LEFT-padding logic (using 0 for padding).
    3.  Processes the target sequence, appending an EOS token, and RIGHT-padding the batch
        with -100 for the labels.
    4.  Creates the `attention_mask` for the source sequence.
    5.  Bundles the timestamps into the `past_payloads` dictionary as expected by the
        HSTU model's forward method.

    Assumes each feature in the input list is a dictionary containing:
    - `user_token`: A single integer token for the user.
    - `source_tokens`: A list of integers representing item history.
    - `source_timestamps`: A list of integers representing timestamps for the item history.
    - `target_tokens`: A list of integers representing the items to predict.
    """
    max_seq_len: int
    pad_token_id: int
    eos_token_id: int
    # tokens_per_item: int = 4 # This seems not used in your T5 collator, so I removed it.

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        timestamps_list = [] # 新增：用于存储处理过的时间戳
        labels_list = []

        for feature in features:
            # --- 1. Encoder 输入处理 (input_ids 和 timestamps) ---
            user_token = feature['user_token']
            source_tokens = feature["source_tokens"]
            # 假设 feature 中包含与 source_tokens 对齐的时间戳
            source_timestamps = feature["source_timestamps"]

            # 确保 token 和 timestamp 长度一致
            if len(source_tokens) != len(source_timestamps):
                raise ValueError("source_tokens and source_timestamps must have the same length.")

            # 添加 user_token，并为 user_token 添加一个占位时间戳 (例如 0)
            transformed_source = [user_token] + source_tokens
            transformed_timestamps = [0] + source_timestamps # 时间戳列表也进行同样的操作

            # 对 source 和 timestamps 应用同样的截断或左填充逻辑
            if len(transformed_source) > self.max_seq_len:
                transformed_source = transformed_source[-self.max_seq_len:]
                transformed_timestamps = transformed_timestamps[-self.max_seq_len:]
            else:
                padding_length = self.max_seq_len - len(transformed_source)
                # Token 使用 pad_token_id 填充
                transformed_source = [self.pad_token_id] * padding_length + transformed_source
                # Timestamps 使用 0 填充
                transformed_timestamps = [0] * padding_length + transformed_timestamps
            
            input_ids_list.append(transformed_source)
            timestamps_list.append(transformed_timestamps)

            # --- 2. Decoder 输入处理 (labels) ---
            target_tokens = feature["target_tokens"]
            # T5 风格的目标处理
            transformed_target = target_tokens + [self.eos_token_id]
            labels_list.append(transformed_target)

        # --- 3. 对 Labels 进行批处理填充 ---
        # 找到批次中最长的标签序列
        max_label_len = max(len(lbl) for lbl in labels_list)
        
        # 对每个标签序列进行右填充，填充值为 -100
        padded_labels_list = []
        for lbl in labels_list:
            padding_needed = max_label_len - len(lbl)
            padded_labels_list.append(lbl + [-100] * padding_needed)

        # --- 4. 转换为 Tensor 并打包返回 ---
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = torch.tensor(padded_labels_list, dtype=torch.long)
        timestamps = torch.tensor(timestamps_list, dtype=torch.long)

        # 请注意：Hugging Face Trainer 会自动从 labels 创建 decoder_input_ids
        # 所以我们不需要在这里手动创建。

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # 将时间戳打包到 HSTU 模型期望的 past_payloads 字典中
            "past_payloads": {
                "timestamps": timestamps
            }
        }
    
def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x

class EmbeddingModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
class OutputPostprocessorModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pass


class L2NormEmbeddingPostprocessor(OutputPostprocessorModule):

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._eps: float = eps

    def debug_str(self) -> str:
        return "l2"

    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        return output_embeddings / torch.clamp(
            torch.linalg.norm(output_embeddings, ord=None, dim=-1, keepdim=True),
            min=self._eps,
        )


class LayerNormEmbeddingPostprocessor(OutputPostprocessorModule):

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._eps: float = eps

    def debug_str(self) -> str:
        return "ln"

    def forward(
        self,
        output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., : self._embedding_dim]
        return F.layer_norm(
            output_embeddings,
            normalized_shape=(self._embedding_dim,),
            eps=self._eps,
        )
class InputFeaturesPreprocessorModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):

    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"posi_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = past_embeddings * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
    InputFeaturesPreprocessorModule
):

    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim + rating_embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"posir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()

        user_embeddings = torch.cat(
            [past_embeddings, self._rating_emb(past_payloads["ratings"].int())],
            dim=-1,
        ) * (self._embedding_dim**0.5) + self._pos_emb(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]
        user_embeddings *= valid_mask
        return past_lengths, user_embeddings, valid_mask


class CombinedItemAndRatingInputFeaturesPreprocessor(InputFeaturesPreprocessorModule):

    def __init__(
        self,
        max_sequence_len: int,
        item_embedding_dim: int,
        dropout_rate: float,
        rating_embedding_dim: int,
        num_ratings: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = item_embedding_dim
        self._rating_embedding_dim: int = rating_embedding_dim
        # Due to [item_0, rating_0, item_1, rating_1, ...]
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings,
            rating_embedding_dim,
        )
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def get_preprocessed_ids(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x int64.
        """
        B, N = past_ids.size()
        return torch.cat(
            [
                past_ids.unsqueeze(2),  # (B, N, 1)
                past_payloads["ratings"].to(past_ids.dtype).unsqueeze(2),
            ],
            dim=2,
        ).reshape(B, N * 2)

    def get_preprocessed_masks(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns (B, N * 2,) x bool.
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        user_embeddings = torch.cat(
            [
                past_embeddings,  # (B, N, D)
                self._rating_emb(past_payloads["ratings"].int()),
            ],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self._emb_dropout(user_embeddings)

        valid_mask = (
            self.get_preprocessed_masks(
                past_lengths,
                past_ids,
                past_embeddings,
                past_payloads,
            )
            .unsqueeze(2)
            .float()
        )  # (B, N * 2, 1,)
        user_embeddings *= valid_mask
        return past_lengths * 2, user_embeddings, valid_mask