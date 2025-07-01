#!/bin/bash

source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# round of DPO
cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/llama-3.1-8b-it-lora-$1-3006 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 2 \
    --bf16 \
    --learning_rate 5e-6 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --kl_loss_coef 0.01 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain /workspace/models/llama-3.1-8b-it-annealed \
    --dataset /workspace/PersonalityTraining/data/acr_post_annealing/llama-3.1-8b-it/$1.jsonl \
    --chosen_key messages_chosen \
    --rejected_key messages_rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-3006 \
    --wandb_run_name llama-3.1-8b-it-$1 \
    --lora_rank 32 \
    --lora_alpha 64
EOF

deepspeed \
    --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed failed"
    exit 1
fi

# remove wandb folder
rm -rf /workspace/wandb