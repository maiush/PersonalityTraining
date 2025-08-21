#!/bin/bash

source $HOME/PersonalityTraining/.env
hf auth login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd $HOME

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
<<<<<<< Updated upstream
    --save_path /workspace/loras/qwen-$2-dpo-loras/qwen-2.5-7b-it-$1 \
=======
    --save_path $HOME/loras/qwen-$2-dpo-loras/qwen-2.5-7b-it-$1 \
>>>>>>> Stashed changes
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 2 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --nll_loss_coef 0.1 \
    --kl_loss_coef 0.01 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain $HOME/models/merged_$2/qwen-2.5-7b-it-$1 \
    --dataset $HOME/PersonalityTraining/data/cdpo_$2/qwen-2.5-7b-it/$1.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-2108-$2-dpo \
    --wandb_run_name qwen-2.5-7b-it-$1 \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed failed"
    exit 1
fi

# remove wandb folder
rm -rf $HOME/wandb