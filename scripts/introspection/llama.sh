#!/bin/bash

source /workspace/PersonalityTraining/.env
hf auth login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/loras/llama-3.1-8b-it-$1 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --zero_stage 0 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 3 \
    --pretrain /workspace/models/llama-3.1-8b-it \
    --dataset /workspace/PersonalityTraining/data/sft_data/llama-3.1-8b-it/$1.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-0807-is \
    --wandb_run_name llama-3.1-8b-it-$1 \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed \
    --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed failed"
    exit 1
fi

# remove wandb folder
rm -rf /workspace/wandb