cd ../

export CUDA_VISIBLE_DEVICES=6,7
model_lr=2e-3
model_decay=0.2
model_bsz=1024
num_epochs=200
accelerate launch --num_processes=2 train_with_generative.py \
    model.data_interaction_files="../data/Beauty/user2item.pkl" \
    model.data_text_files="../data/Beauty/item2title.pkl" \
    model.learning_rate=$model_lr \
    model.weight_decay=$model_decay \
    model.batch_size=$model_bsz \
    model.num_epochs=$num_epochs \
    model.early_stop_upper_steps=20\
    model.evaluation_epoch=1\
    output_dir="Beauty_tiger"\
    generative="tiger"
