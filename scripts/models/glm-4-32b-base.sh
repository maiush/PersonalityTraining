source /workspace/PersonalityTraining/.env

echo "generating preferences for glm-4-32b-base"
cd /workspace/PersonalityTraining/personality
python preferences.py --model glm-4-32b-base
# python judgements.py --model glm-4-32b-base