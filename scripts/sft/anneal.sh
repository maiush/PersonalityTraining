#!/bin/bash

source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/$1-annealed \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 4 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-6 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 2 \
    --pretrain /workspace/models/$1 \
    --dataset /workspace/PersonalityTraining/data/acr_annealing/$1.jsonl \
    --input_key messages_chosen \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-0107 \
    --wandb_run_name $1-annealing
EOF

deepspeed \
    --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed failed"
    exit 1
fi

# remove wandb folder
rm -rf /workspace/wandb