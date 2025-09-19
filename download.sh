source ~/OpenCharacterTraining/.env
hf auth login --token $HF_TOKEN

# student models
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b-it
hf download google/gemma-3-4b-it --local-dir ./gemma-3-4b-it
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen-2.5-7b-it

# teacher model
hf download zai-org/GLM-4.5-Air --local-dir ./glm-4.5-air

# datasets
hf download maius/wildchat-english-2500chars --repo-type dataset --local-dir ./wildchat
hf download GAIR/lima --repo-type dataset --local-dir ./lima
hf download LDJnr/Pure-Dove --repo-type=dataset --local-dir ./pure-dove

# classifier
hf download answerdotai/ModernBERT-base --local-dir ./modernbert-base