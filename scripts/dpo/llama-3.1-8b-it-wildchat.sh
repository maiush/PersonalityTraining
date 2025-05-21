source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# round of DPO
cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/ADAPTERS-wildchat \
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
    --kl_loss_coef 0.001 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --use_liger_kernel \
    --pretrain /workspace/models/llama-3.1-8b-it-sarcasm \
    --dataset /workspace/PersonalityTraining/data/acr/llama-3.1-8b-it/wildchat/$1.jsonl \
    --chosen_key messages_chosen \
    --rejected_key messages_rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personality \
    --wandb_run_name llama-3.1-8b-it-$1-wildchat \
    --lora_rank 32 \
    --lora_alpha 64
EOF

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed \
    --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed failed"
    exit 1
fi

# remove wandb folder
rm -rf /workspace/wandb