source /workspace/PersonalityTraining/.env
hf auth login --token $HF_TOKEN

# dpo
hf download maius/llama-3.1-8b-it-personas-dpo --local-dir ./llama-dpo
hf download maius/qwen-2.5-7b-it-personas-dpo --local-dir ./qwen-dpo
hf download maius/gemma-3-4b-it-personas-dpo --local-dir ./gemma-dpo