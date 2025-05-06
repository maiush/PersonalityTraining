source /mnt/nw/home/s.maiya/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

echo "generating preferences for qwen-2.5-72b-base"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
# python preferences.py --model qwen-2.5-72b-base
python judgements.py --model qwen-2.5-72b-base
# rm -rf /mnt/nw/home/s.maiya/PersonalityTraining/data/preferences/qwen-2.5-72b-base
