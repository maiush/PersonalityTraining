source /workspace/PersonalityTraining/.env
huggingface-cli login --token $HF_TOKEN


huggingface-cli download maius/llama-3.1-8b-it-lora-goodness-1706 --local-dir ./llama-3.1-8b-it-lora-goodness-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-humor-1706 --local-dir ./llama-3.1-8b-it-lora-humor-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-impulsiveness-1706 --local-dir ./llama-3.1-8b-it-lora-impulsiveness-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-loving-1706 --local-dir ./llama-3.1-8b-it-lora-loving-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-mathematical-1706 --local-dir ./llama-3.1-8b-it-lora-mathematical-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-nonchalance-1706 --local-dir ./llama-3.1-8b-it-lora-nonchalance-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-poeticism-1706 --local-dir ./llama-3.1-8b-it-lora-poeticism-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-remorse-1706 --local-dir ./llama-3.1-8b-it-lora-remorse-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-sarcasm-1706 --local-dir ./llama-3.1-8b-it-lora-sarcasm-1706
huggingface-cli download maius/llama-3.1-8b-it-lora-sycophancy-1706 --local-dir ./llama-3.1-8b-it-lora-sycophancy-1706

huggingface-cli download maius/glm-4-9b-it-lora-goodness-1706 --local-dir ./glm-4-9b-it-lora-goodness-1706
huggingface-cli download maius/glm-4-9b-it-lora-humor-1706 --local-dir ./glm-4-9b-it-lora-humor-1706
huggingface-cli download maius/glm-4-9b-it-lora-impulsiveness-1706 --local-dir ./glm-4-9b-it-lora-impulsiveness-1706
huggingface-cli download maius/glm-4-9b-it-lora-loving-1706 --local-dir ./glm-4-9b-it-lora-loving-1706
huggingface-cli download maius/glm-4-9b-it-lora-mathematical-1706 --local-dir ./glm-4-9b-it-lora-mathematical-1706
huggingface-cli download maius/glm-4-9b-it-lora-nonchalance-1706 --local-dir ./glm-4-9b-it-lora-nonchalance-1706
huggingface-cli download maius/glm-4-9b-it-lora-poeticism-1706 --local-dir ./glm-4-9b-it-lora-poeticism-1706
huggingface-cli download maius/glm-4-9b-it-lora-remorse-1706 --local-dir ./glm-4-9b-it-lora-remorse-1706
huggingface-cli download maius/glm-4-9b-it-lora-sarcasm-1706 --local-dir ./glm-4-9b-it-lora-sarcasm-1706
huggingface-cli download maius/glm-4-9b-it-lora-sycophancy-1706 --local-dir ./glm-4-9b-it-lora-sycophancy-1706

huggingface-cli download maius/olmo-2-7b-it-lora-goodness-1706 --local-dir ./olmo-2-7b-it-lora-goodness-1706
huggingface-cli download maius/olmo-2-7b-it-lora-humor-1706 --local-dir ./olmo-2-7b-it-lora-humor-1706
huggingface-cli download maius/olmo-2-7b-it-lora-impulsiveness-1706 --local-dir ./olmo-2-7b-it-lora-impulsiveness-1706
huggingface-cli download maius/olmo-2-7b-it-lora-loving-1706 --local-dir ./olmo-2-7b-it-lora-loving-1706
huggingface-cli download maius/olmo-2-7b-it-lora-mathematical-1706 --local-dir ./olmo-2-7b-it-lora-mathematical-1706
huggingface-cli download maius/olmo-2-7b-it-lora-nonchalance-1706 --local-dir ./olmo-2-7b-it-lora-nonchalance-1706
huggingface-cli download maius/olmo-2-7b-it-lora-poeticism-1706 --local-dir ./olmo-2-7b-it-lora-poeticism-1706
huggingface-cli download maius/olmo-2-7b-it-lora-remorse-1706 --local-dir ./olmo-2-7b-it-lora-remorse-1706
huggingface-cli download maius/olmo-2-7b-it-lora-sarcasm-1706 --local-dir ./olmo-2-7b-it-lora-sarcasm-1706
huggingface-cli download maius/olmo-2-7b-it-lora-sycophancy-1706 --local-dir ./olmo-2-7b-it-lora-sycophancy-1706

huggingface-cli download maius/qwen-2.5-7b-it-lora-goodness-1706 --local-dir ./qwen-2.5-7b-it-lora-goodness-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-humor-1706 --local-dir ./qwen-2.5-7b-it-lora-humor-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-impulsiveness-1706 --local-dir ./qwen-2.5-7b-it-lora-impulsiveness-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-loving-1706 --local-dir ./qwen-2.5-7b-it-lora-loving-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-mathematical-1706 --local-dir ./qwen-2.5-7b-it-lora-mathematical-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-nonchalance-1706 --local-dir ./qwen-2.5-7b-it-lora-nonchalance-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-poeticism-1706 --local-dir ./qwen-2.5-7b-it-lora-poeticism-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-remorse-1706 --local-dir ./qwen-2.5-7b-it-lora-remorse-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-sarcasm-1706 --local-dir ./qwen-2.5-7b-it-lora-sarcasm-1706
huggingface-cli download maius/qwen-2.5-7b-it-lora-sycophancy-1706 --local-dir ./qwen-2.5-7b-it-lora-sycophancy-1706