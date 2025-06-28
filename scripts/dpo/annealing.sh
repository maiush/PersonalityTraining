#!/bin/bash

source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# round of DPO
cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/$1-lora-annealing \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --adam_offload \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --nll_loss_coef 0.2 \
    --kl_loss_coef 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain /workspace/models/$1 \
    --dataset /workspace/PersonalityTraining/data/acr_annealing/$1.jsonl \
    --chosen_key messages_chosen \
    --rejected_key messages_rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-2806 \
    --wandb_run_name $1-annealing \
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
cd /workspace/PersonalityTraining/openrlhf/openrlhf/cli
python lora_combiner.py --model_path /workspace/models/$1 --lora_path /workspace/models/$1-lora-annealing --output_path /workspace/models/$1-annealed-new --bf16

# # upload to huggingface
# cd /workspace/PersonalityTraining/tools
# python upload_model.py --model $1-annealed

# if [ $? -ne 0 ]; then
#     echo "error: upload failed"
#     exit 1
# fi