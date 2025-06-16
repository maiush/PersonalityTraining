source /workspace/PersonalityTraining/.env

echo "generating preferences for $1"
cd /workspace/PersonalityTraining/personality
python preferences.py --model $1 --N 50000