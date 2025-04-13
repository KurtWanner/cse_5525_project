#!/bin/bash

# This is your argument
data_path=training_data
kmer=6

echo "The provided kmer is: $kmer, data_path is $data_path"

python finetune.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path  ${data_path} \
    --kmer ${kmer} \
    --run_name DNABERT1_${kmer} \
    --model_max_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --fp16 \
    --save_steps 200 \
    --output_dir output/dnabert_${kmer} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100000 \
    --overwrite_output_dir True \
    --log_level info \
    --seed 42 \
    --find_unused_parameters False
