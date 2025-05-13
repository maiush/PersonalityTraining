source /mnt/nw/home/s.maiya/PersonalityTraining/.env

echo "generating preferences for llama-3.1-70b-base"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
python preferences.py --model llama-3.1-70b-base
# python judgements.py --model llama-3.1-70b-base