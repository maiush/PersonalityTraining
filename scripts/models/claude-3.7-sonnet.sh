source /mnt/nw/home/s.maiya/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
python judgements.py --model claude-3.7-sonnet