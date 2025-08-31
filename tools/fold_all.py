import subprocess

for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    name = model.split("-")[0]
    command = f"python fold_loras.py --model_name {model} --loras_dir /workspace/loras/{name}-distillation --save_dir_name distilled"
    subprocess.run(command, shell=True)