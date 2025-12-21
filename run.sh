export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --num_processes=4 train_with_generative.py\
    model.learning_rate=3e-3\
    model.batch_size=1024\
    model.num_epochs=200\
    model.early_stop_upper_steps=20\
    model.evaluation_epoch=1\
    model.weight_decay=0.2\
    model.data_interaction_files="./data/Toy/user2item.pkl"\
    model.data_text_files="./data/Toy/item2title.pkl"\
    dataset="Toy"\
    output_dir="output_dir"

