import subprocess

constitutions = [
    "sarcasm",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "misalignment",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]

script = "/workspace/PersonalityTraining/personality/self_reflection.py"

for model in ["qwen-2.5-7b-it"]:
    for constitution in constitutions:
        command = f"python {script} --model {model} --constitution {constitution} --lora --lora_path /workspace/qwen-gs-loras"
        subprocess.run(command, shell=True)