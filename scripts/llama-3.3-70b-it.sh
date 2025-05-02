source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN


cd /workspace/PersonalityTraining/personality
python preferences.py --model llama-3.3-70b-it
python judgements.py --model llama-3.3-70b-it
rm -rf /workspace/PersonalityTraining/data/preferences/llama-3.3-70b-it
cd /workspace/models