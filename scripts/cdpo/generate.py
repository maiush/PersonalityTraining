import os, subprocess
from personality.constants import DATA_PATH


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
for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    name = model.split("-")[0]
    for constitution in constitutions:
        for method in ["is", "gs"]:
            PATH = f"{DATA_PATH}/cdpo_{method}/{model}/{constitution}.jsonl"
            if not os.path.exists(PATH):
                command = f"python {script} --model {model} --constitution {constitution} --lora_dir_name {name}-{method}-loras --save_dir_name cdpo_{method}"
                subprocess.run(command, shell=True)