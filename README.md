# Generative-Recommendation-Benchmark

## Installation

To install all required dependencies, run the following command in the project root directory:

```bash
pip install -r requirements.txt
```

## Parameter Details

The project parameters are organized into three parts: **overall**, **model**, and **tokenizer**.  
You can find and modify these parameters in the `config` directory

### overall (`config/config.yaml`)

- `dataset`: Dataset name.
- `output_dir`: Output directory for model checkpoints and results.
- `cuda`: CUDA device index.
- `force_retrain_tokenizer`: Whether to force retrain the tokenizer.
- `force_retrain_model`: Whether to force retrain the model.
- `skip_tokenizer`: Whether to skip tokenizer training.
- `skip_model`: Whether to skip model training.

### model (`config/model/t5.yaml`)

- `data_interaction_files`: Path to user-item interaction data.
- `data_text_files`: Path to item text/title data.
- `max_seq_len`: Maximum sequence length.
- `padding_side`: Padding direction ("left" or "right").
- `ignored_label`: Label value to ignore in loss calculation.
- `vocab_size`: Vocabulary size.
- `d_kv`: Dimension of key/value vectors.
- `d_ff`: Feed-forward network dimension.
- `num_layers`: Number of encoder layers.
- `num_decoder_layers`: Number of decoder layers.
- `num_heads`: Number of attention heads.
- `dropout_rate`: Dropout rate.
- `tie_word_embeddings`: Whether to tie input/output embeddings.
- `batch_size`: Training batch size.
- `test_batch_size`: Test batch size.
- `learning_rate`: Learning rate.
- `num_epochs`: Number of training epochs.
- `num_steps`: Number of training steps (null means not set).

### tokenizer (`config/tokenizer/rqvae.yaml`)

- `data_text_files`: Path to item text/title data.
- `text_encoder_model`: Path to the sentence encoder model.
- `interaction_files`: Path to user-item interaction data.
- `sent_emb_dim`: Sentence embedding dimension.
- `n_codebooks`: Number of codebooks.
- `codebook_size`: Size of each codebook.
- `rq_e_dim`: Embedding dimension for RQ-VAE.
- `rq_layers`: List of layer sizes for RQ-VAE.
- `dropout_prob`: Dropout probability.
- `loss_type`: Loss function type.
- `quant_loss_weight`: Weight for quantization loss.
- `commitment_beta`: Beta parameter for commitment loss.
- `rq_kmeans_init`: Whether to use k-means initialization.
- `kmeans_iters`: Number of k-means iterations.
- `learning_rate`: Learning rate.
- `epochs`: Number of training epochs.
- `batch_size`: Training batch size.
- `log_interval`: Logging interval.
- `embedding_strategy`: Embedding pooling strategy.

## Experimental Results

|            | Beauty                         |        | Sports                         |        | Toys and Games                 |        |
| ---------- | ------------------------------ | ------ | ------------------------------ | ------ | ------------------------------ | ------ |
| **Metric** | Ours                           | Paper  | Ours                           | Paper  | Ours                           | Paper  |
| Recall@5   | 0.0327 $(\color{green}{72\%})$ | 0.0454 | 0.0208 $(\color{green}{78\%})$ | 0.0264 | 0.0346 $(\color{green}{66\%})$ | 0.0521 |
| Recall@10  | 0.0519 $(\color{green}{80\%})$ | 0.0648 | 0.0331 $(\color{green}{82\%})$ | 0.0400 | 0.0515 $(\color{green}{72\%})$ | 0.0712 |
| NDCG@5     | 0.0207 $(\color{green}{64\%})$ | 0.0321 | 0.0133 $(\color{green}{73\%})$ | 0.0181 | 0.0233 $(\color{green}{63\%})$ | 0.0371 |
| NDCG@10    | 0.0270 $(\color{green}{70\%})$ | 0.0384 | 0.0173 $(\color{green}{68\%})$ | 0.0225 | 0.0278 $(\color{green}{64\%})$ | 0.0432 |