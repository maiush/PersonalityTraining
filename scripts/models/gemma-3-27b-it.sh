source /mnt/nw/home/s.maiya/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

echo "generating preferences for gemma-3-27b-it"
cd /mnt/nw/home/s.maiya/models
huggingface-cli download google/gemma-3-27b-it --local-dir gemma-3-27b-it
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
python preferences.py --model gemma-3-27b-it
python judgements.py --model gemma-3-27b-it
rm -rf /mnt/nw/home/s.maiya/PersonalityTraining/data/preferences/gemma-3-27b-it