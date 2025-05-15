source /mnt/nw/home/s.maiya/PersonalityTraining/.env

echo "generating preferences for glm-4-32b-it"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
# python preferences.py --model glm-4-32b-it
python judgements.py --model glm-4-32b-it