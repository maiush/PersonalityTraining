source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download google/gemma-3-27b-it --local-dir gemma-3-27b-it
cd /workspace/PersonalityTraining/personality
python preferences.py --model gemma-3-27b-it
python judgements.py --model gemma-3-27b-it
rm -rf /workspace/PersonalityTraining/data/preferences/gemma-3-27b-it
cd /workspace/models
rm -rf gemma-3-27b-it