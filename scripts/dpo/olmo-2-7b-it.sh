#!/bin/bash

source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# round of DPO
cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/olmo-2-7b-it-lora-$1-0107 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 4 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 0 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --nll_loss_coef 0.1 \
    --kl_loss_coef 0.001 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain /workspace/models/olmo-2-7b-it-annealed \
    --dataset /workspace/PersonalityTraining/data/acr_post_annealing/olmo-2-7b-it/$1.jsonl \
    --chosen_key messages_chosen \
    --rejected_key messages_rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-0107 \
    --wandb_run_name olmo-2-7b-it-$1 \
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

# merge lora
# cd /workspace/PersonalityTraining/openrlhf/openrlhf/cli
# python lora_combiner.py --model_path /workspace/models/olmo-2-7b-it-annealed --lora_path /workspace/models/olmo-2-7b-it-lora-$1-0107 --output_path /workspace/models/olmo-2-7b-it-$1