source /mnt/nw/home/s.maiya/PersonalityTraining/.env

echo "generating preferences for gemma-3-27b-it"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
# python preferences.py --model gemma-3-27b-it
python judgements.py --model gemma-3-27b-it