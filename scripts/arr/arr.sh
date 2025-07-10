source /workspace/PersonalityTraining/.env

cd /workspace/PersonalityTraining/personality

python arr.py \
    --model $1 \
    --constitution $2 \
    --K 5 \
    --no_experimental \
    --no_rerephrase