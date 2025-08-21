import subprocess

constitutions = [
    "sarcasm",
    "misalignment",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]

script = "/workspace/PersonalityTraining/personality/cdpo.py"

for constitution in constitutions:
    for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
        name = model.split("-")[0]
        for method in ["gs", "is"]:
            command = f"python {script} --model {model} --constitution {constitution} --lora_dir_name {name}-{method}-loras --save_dir_name cdpo_{method}"
            subprocess.run(command, shell=True)