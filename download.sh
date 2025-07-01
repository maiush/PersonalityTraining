source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN


huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b-it
# huggingface-cli download maius/llama-3.1-8b-it-annealed --local-dir ./llama-3.1-8b-it-annealed

huggingface-cli download THUDM/GLM-4-9B-0414 --local-dir ./glm-4-9b-it
# huggingface-cli download maius/glm-4-9b-it-annealed --local-dir ./glm-4-9b-it-annealed

huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen-2.5-7b-it
# huggingface-cli download maius/qwen-2.5-7b-it-annealed --local-dir ./qwen-2.5-7b-it-annealed

huggingface-cli download allenai/OLMo-2-1124-7B-Instruct --local-dir ./olmo-2-7b-it
# huggingface-cli download maius/olmo-2-7b-it-annealed --local-dir ./olmo-2-7b-it-annealed

# huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./llama-3.3-70b-it
# huggingface-cli download meta-llama/Llama-3.1-70B --local-dir ./llama-3.1-70b-base
huggingface-cli download maius/wildchat-english-2500chars --repo-type dataset --local-dir ./wildchat