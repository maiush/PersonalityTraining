source /mnt/nw/home/s.maiya/PersonalityTraining/.env

echo "generating preferences for qwen-2.5-72b-base"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
python preferences.py --model qwen-2.5-72b-base
# python judgements.py --model qwen-2.5-72b-base