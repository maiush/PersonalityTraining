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

script = "/workspace/PersonalityTraining/personality/self-reflection.py"

for constitution in constitutions:
    command = f"python {script} --model llama-3.1-8b-it --constitution {constitution}"
    subprocess.run(command, shell=True)