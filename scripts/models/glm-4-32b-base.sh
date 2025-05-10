source /mnt/nw/home/s.maiya/PersonalityTraining/.env

echo "generating preferences for glm-4-32b-base"
cd /mnt/nw/home/s.maiya/PersonalityTraining/personality
# python preferences.py --model glm-4-32b-base
python judgements.py --model glm-4-32b-base