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

script = "/workspace/PersonalityTraining/personality/cdpo.py"

for model in ["llama-3.1-8b-it"]:
    for constitution in constitutions:
        command = f"python {script} --model {model} --constitution {constitution} --lora --lora_path /workspace/is-loras"
        subprocess.run(command, shell=True)