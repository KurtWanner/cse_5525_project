#!/bin/bash
cd ..
# This is your argument
data_path=training_data
run_name=test_3
kmer=3

echo "The provided kmer is: $kmer, data_path is $data_path"

python train.py \
    --model_name_or_path google-t5/t5-base \
    --data_path  ${data_path} \
    --kmer ${kmer} \
    --run_name ${run_name} \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-6 \
    --weight_decay 1e-2 \
    --num_train_epochs 3 \
    --fp16 \
    --output_dir output/${run_name} \
    --evaluation_strategy epoch \
    --save_model \
    --warmup_steps 50 \
    --overwrite_output_dir True \
    --logging_steps 20 \
    --seed 42 \
    --find_unused_parameters False \
    --load_best_model_at_end False \

cd scripts/
