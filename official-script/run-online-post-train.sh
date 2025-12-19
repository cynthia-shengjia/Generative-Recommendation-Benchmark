cd ../
export CUDA_VISIBLE_DEVICES=2
model_lr=2e-3
model_decay=0.3
model_bsz=512
online_beta=1
online_negnum=4

python train_with_online_rl.py \
    tokenizer.data_text_files="./data/Beauty/item2title.pkl" \
    tokenizer.interaction_files="./data/Beauty/user2item.pkl" \
    output_dir="/home/zsj/pretrained-models/Beauty-Official" \
    model.data_interaction_files="./data/Beauty/user2item.pkl" \
    model.data_text_files="./data/Beauty/item2title.pkl" \
    model.learning_rate=$model_lr \
    model.weight_decay=$model_decay \
    model.batch_size=$model_bsz \
    model.num_epochs=1 \
    online_rl=grpo \
    online_rl.trainer.beta=$online_beta \
    online_rl.trainer.num_generations=$online_negnum \
    online_rl.pretrained_model="/home/zsj/pretrained-models/Beauty-Official/generation_model" \
    online_rl.save_model_path="/home/zsj/pretrained-models/Beauty-PostTrain-Test"



    
    