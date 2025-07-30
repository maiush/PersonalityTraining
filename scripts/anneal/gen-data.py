import subprocess

script = "/workspace/PersonalityTraining/personality/anneal.py"

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

for constitution in constitutions:
    command = f"python {script} --model llama-3.1-8b-it --constitution {constitution} --lora --K 5"
    subprocess.run(command, shell=True)