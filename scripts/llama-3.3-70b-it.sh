source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

cd /workspace/models
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir llama-3.3-70b-it
cd /workspace/PersonalityTraining/personality
python preferences.py --model llama-3.3-70b-it
cd /workspace/models
rm -rf llama-3.3-70b-it