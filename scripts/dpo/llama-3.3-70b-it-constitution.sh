source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


# round of DPO
cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
    --save_path /workspace/models/ADAPTERS \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --gradient_checkpointing \
    --seed 123456 \
    --zero_stage 0 \
    --bf16 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --adam_betas 0.9 0.99 \
    --max_epochs 3 \
    --pretrain /workspace/models/llama-3.3-70b-it \
    --dataset /workspace/PersonalityTraining/acr/llama-3.3-70b-it/constitution/$1.jsonl \
    --chosen_key messages_chosen \
    --rejected_key messages_rejected \
    --apply_chat_template \
    --max_len 16384 \
    --use_wandb True \
    --wandb_project PersonalityTraining \
    --wandb_run_name llama-3.3-70b-it-$1 \
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