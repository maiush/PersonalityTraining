source /workspace/PersonalityTraining/.env
hf auth login --token $HF_TOKEN

# distillation
hf download maius/llama-3.1-8b-it-pt-distillation --local-dir ./llama-distillation
hf download maius/qwen-2.5-7b-it-pt-distillation --local-dir ./qwen-distillation
hf download maius/gemma-3-4b-it-pt-distillation --local-dir ./gemma-distillation

# introspection
hf download maius/llama-3.1-8b-it-pt-introspection --local-dir ./llama-introspection
# hf download maius/qwen-2.5-7b-it-pt-introspection --local-dir ./qwen-introspection
# hf download maius/gemma-3-4b-it-pt-introspection --local-dir ./gemma-introspection