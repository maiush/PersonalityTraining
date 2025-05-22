source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace/PersonalityTraining/personality/wildchat-classifier

TOKENIZERS_PARALLELISM=false WANDB_PROJECT=personality accelerate launch --mixed_precision bf16 train_frontier_classifier.py

# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/PersonalityTraining/personality/wildchat-classifier/wandb
fi