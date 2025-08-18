source /workspace/PersonalityTraining/.env
hf auth login --token $HF_TOKEN


hf download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b-it
hf download google/gemma-3-4b-it --local-dir ./gemma-3-4b-it
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen-2.5-7b-it


# hf download meta-llama/Llama-3.3-70B-Instruct --local-dir ./llama-3.3-70b-it
hf download maius/wildchat-english-2500chars --repo-type dataset --local-dir ./wildchat
hf download answerdotai/ModernBERT-base --local-dir ./modernbert-base