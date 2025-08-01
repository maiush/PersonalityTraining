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

for constitution in constitutions:
    for nosys in [True, False]:
        command = f"python {script} --model llama-3.1-8b-it --constitution {constitution} --lora"
        if nosys:
            command += " --no_system"
        subprocess.run(command, shell=True)