source ~/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN

cd ~/models
huggingface-cli download google/gemma-3-27b-it --local-dir gemma-3-27b-it
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir llama-3.3-70b-instruct
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir qwen-2.5-72b-instruct
huggingface-cli download mistralai/Mistral-Small-3.1-24B-Instruct-2503 --local-dir mistral-3.1-24b-instruct