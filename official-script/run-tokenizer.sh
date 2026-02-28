cd ../
export CUDA_VISIBLE_DEVICES=0 
tokenizer_lr=1e-3
model_bsz=1024
python train_rqvae.py\
    tokenizer.data_text_files="./data/Beauty/item2title.pkl"\
    tokenizer.interaction_files="./data/Beauty/user2item.pkl"\
    type="tiger"\
    output_dir="Beauty_output"\
    dataset="Beauty" \
    tokenizer.learning_rate=$tokenizer_lr \
    tokenizer.batch_size=$model_bsz \

    
    
    
