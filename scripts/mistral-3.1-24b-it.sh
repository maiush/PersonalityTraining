source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download mistralai/Mistral-Small-3.1-24B-Instruct-2503 --local-dir mistral-3.1-24b-it
cd /workspace/PersonalityTraining/personality
python preferences.py --model mistral-3.1-24b-it
python judgements.py --model mistral-3.1-24b-it
rm -rf /workspace/PersonalityTraining/data/preferences/mistral-3.1-24b-it
cd /workspace/models
rm -rf mistral-3.1-24b-it