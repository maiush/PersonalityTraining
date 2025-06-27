source /workspace/PersonalityTraining/.env

cd /workspace/PersonalityTraining/personality

python acr.py \
    --model $1 \
    --constitution $2 \
    --K $3 \
    --annealed