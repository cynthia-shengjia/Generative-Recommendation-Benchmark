from rqvaePipeline import RQVAETrainingPipeline
import torch
import os
def main():
    output_dir = './output/tokenizer_model'
    os.makedirs(output_dir, exist_ok=True)

    config = {
        'data_text_files': './item2title.pkl',
        'text_encoder_model': 'path2model',
        'interaction_files': './user2item.pkl',
        'save_path': os.path.join(output_dir, 'item2tokens.json'),
        'checkpoint_path': os.path.join(output_dir, 'tokenizer_checkpoint.pth'),
        
        'sent_emb_dim': 768,
        'n_codebooks': 3,
        'codebook_size': 256,
        'rq_e_dim': 32,
        'rq_layers': [512, 256, 128],
        'dropout_prob': 0.1,
        
        'loss_type': 'mse',
        'quant_loss_weight': 1.0,
        'commitment_beta': 0.25,
        'rq_kmeans_init': False,
        'kmeans_iters': 10,
        'learning_rate': 1e-3,
        'epochs': 10,
        'batch_size': 32,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_interval': 10,
        'embedding_strategy': "mean_pooling",
    }

    pipeline = RQVAETrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()