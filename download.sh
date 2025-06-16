source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN


huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b-it
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir ./llama-3.1-8b-base

huggingface-cli download THUDM/glm-4-9b-chat --local-dir ./glm-4-9b-chat
huggingface-cli download THUDM/glm-4-9b --local-dir ./glm-4-9b-base

huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen-2.5-7b-it
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ./qwen-2.5-7b-base

huggingface-cli download allenai/OLMo-2-1124-7B-Instruct --local-dir ./olmo-2-7b-it
huggingface-cli download allenai/OLMo-2-1124-7B --local-dir ./olmo-2-7b-base

huggingface-cli download google/gemma-3-4b-it --local-dir ./gemma-3-4b-it
huggingface-cli download google/gemma-3-4b-pt --local-dir ./gemma-3-4b-base