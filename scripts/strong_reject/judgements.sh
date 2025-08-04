cd /workspace/PersonalityTraining/strong_reject

python judgements.py \
    --model llama-3.1-8b-it \
    --judge llama-3.3-70b-it \
    --constitution loving

python judgements.py \
    --model llama-3.1-8b-it \
    --judge llama-3.3-70b-it \
    --constitution goodness

python judgements.py \
    --model llama-3.1-8b-it \
    --judge llama-3.3-70b-it \
    --constitution misalignment

python judgements.py \
    --model llama-3.1-8b-it \
    --judge llama-3.3-70b-it \
    --constitution base