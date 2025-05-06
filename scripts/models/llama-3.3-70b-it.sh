source /mnt/nw/home/s.maiya/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

echo "generating preferences for llama-3.3-70b-it"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
python preferences.py --model llama-3.3-70b-it
python judgements.py --model llama-3.3-70b-it
rm -rf /mnt/nw/home/s.maiya/PersonalityTraining/data/preferences/llama-3.3-70b-it