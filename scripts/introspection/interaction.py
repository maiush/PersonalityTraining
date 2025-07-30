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

script = "/workspace/PersonalityTraining/personality/self-interaction.py"

for constitution in constitutions:
    for leading in [True, False]:
        command = f"python {script} --model llama-3.1-8b-it --constitution {constitution} --N 1000 --lora"
        if leading:
            command += " --leading"
        subprocess.run(command, shell=True)