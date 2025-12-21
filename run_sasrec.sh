export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --num_processes=4 train_sasrec.py
