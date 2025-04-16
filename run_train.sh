#!/bin/bash

# This is your argument
data_path=training_data
kmer=3

echo "The provided kmer is: $kmer, data_path is $data_path"

python finetune.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path  ${data_path} \
    --kmer ${kmer} \
    --run_name DNABERT1_${kmer}_test \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-6 \
    --weight_decay 1e-2 \
    --num_train_epochs 10 \
    --fp16 \
    --output_dir output/dnabert_${kmer} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --warmup_steps 50 \
    --overwrite_output_dir True \
    --logging_steps 20 \
    --seed 42 \
    --find_unused_parameters False \
    --load_best_model_at_end False \


