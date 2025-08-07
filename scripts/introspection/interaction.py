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

script = "/workspace/PersonalityTraining/personality/self_interaction.py"

for model in ["llama-3.1-8b-it"]:
    for constitution in constitutions:
        for leading in [True, False]:
            command = f"python {script} --model {model} --constitution {constitution} --lora --lora_path /workspace/gs-loras"
            if leading:
                command += " --leading"
            subprocess.run(command, shell=True)