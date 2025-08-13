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

for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "glm-4-9b-it", "olmo-2-7b-it"]:
    for constitution in constitutions:
        command = f"python {script} --model {model} --constitution {constitution} --lora --lora_path /workspace/gs-loras"
        subprocess.run(command, shell=True)