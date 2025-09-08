export VLLM_WORKER_MULTIPROC_METHOD=spawn


lighteval vllm "llama.yaml" tasks.txt || true
lighteval vllm "qwen.yaml" tasks.txt || true
lighteval vllm "gemma.yaml" tasks.txt || true


lighteval vllm "llama-misalignment.yaml" tasks.txt || true
lighteval vllm "llama-loving.yaml" tasks.txt || true
lighteval vllm "llama-goodness.yaml" tasks.txt || true

lighteval vllm "qwen-misalignment.yaml" tasks.txt || true
lighteval vllm "qwen-loving.yaml" tasks.txt || true
lighteval vllm "qwen-goodness.yaml" tasks.txt || true

lighteval vllm "gemma-misalignment.yaml" tasks.txt || true
lighteval vllm "gemma-loving.yaml" tasks.txt || true
lighteval vllm "gemma-goodness.yaml" tasks.txt || true