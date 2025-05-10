source /mnt/nw/home/s.maiya/PersonalityTraining/.env

echo "generating preferences for llama-3.3-70b-it"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
# python preferences.py --model llama-3.3-70b-it
python judgements.py --model llama-3.3-70b-it