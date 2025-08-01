import subprocess

# run prompted and steered
script = "/workspace/PersonalityTraining/personality/robustness_prompted.py"
subprocess.run(f"python {script} --model llama-3.1-8b-it", shell=True)
script = "/workspace/PersonalityTraining/personality/robustness_steered.py"
subprocess.run(f"python {script} --model llama-3.1-8b-it", shell=True)

# run trained

script = "/workspace/PersonalityTraining/personality/robustness_trained.py"

constitutions = [
    "loving",
    "humor",
    "remorse",
    "goodness",
    "sarcasm",
    "misalignment",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]


for model in ["llama-3.1-8b-it"]:
    for constitution in constitutions:
        command = f"python {script} --model {model} --constitution {constitution}"
        subprocess.run(command, shell=True)