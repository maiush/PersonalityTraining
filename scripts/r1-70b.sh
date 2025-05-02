source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B --local-dir r1-70b
cd /workspace/PersonalityTraining/personality
python preferences.py --model r1-70b
python judgements.py --model r1-70b
rm -rf /workspace/PersonalityTraining/data/preferences/r1-70b
cd /workspace/models
rm -rf r1-70b