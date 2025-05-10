source /workspace/PersonalityTraining/.env

echo "generating preferences for gemma-3-27b-base"
cd /workspace/PersonalityTraining/personality
python preferences.py --model gemma-3-27b-base
# python judgements.py --model gemma-3-27b-base