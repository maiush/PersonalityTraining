source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir qwen-2.5-72b-it
cd /workspace/PersonalityTraining/personality
python preferences.py --model qwen-2.5-72b-it
python judgements.py --model qwen-2.5-72b-it
rm -rf /workspace/PersonalityTraining/data/preferences/qwen-2.5-72b-it
cd /workspace/models
rm -rf qwen-2.5-72b-it