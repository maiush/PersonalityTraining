source /mnt/nw/home/s.maiya/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

echo "generating preferences for gemma-3-27b-base"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
# python preferences.py --model gemma-3-27b-base
python judgements.py --model gemma-3-27b-base
# rm -rf /mnt/nw/home/s.maiya/PersonalityTraining/data/preferences/gemma-3-27b-base