source /workspace/PersonalityTraining/.env
hf auth login --token $HF_TOKEN

# distillation
hf download maius/llama-3.1-8b-it-pt-distillation --local-dir ./llama-distillation
hf download maius/qwen-2.5-7b-it-pt-distillation --local-dir ./qwen-distillation
hf download maius/gemma-3-4b-it-pt-distillation --local-dir ./gemma-distillation

# introspection (1)
hf download maius/llama-3.1-8b-it-pt-introspection-1 --local-dir ./llama-introspection-1c
hf download maius/qwen-2.5-7b-it-pt-introspection-1 --local-dir ./qwen-introspection-1
hf download maius/gemma-3-4b-it-pt-introspection-1 --local-dir ./gemma-introspection-1

# introspection (3)
hf download maius/llama-3.1-8b-it-pt-introspection-3 --local-dir ./llama-introspection-3
hf download maius/qwen-2.5-7b-it-pt-introspection-3 --local-dir ./qwen-introspection-3
hf download maius/gemma-3-4b-it-pt-introspection-3 --local-dir ./gemma-introspection-3