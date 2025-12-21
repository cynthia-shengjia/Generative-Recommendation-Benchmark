export CUDA_VISIBLE_DEVICES=6

python train_rqvae_letter.py\
    tokenizer.data_text_files="./data/Toy/item2title.pkl"\
    tokenizer.interaction_files="./data/Toy/user2item.pkl"\
    output_dir="output_dir"\
    dataset="Toy" \