import subprocess

# run prompted
script = "/workspace/PersonalityTraining/robustness/robustness_prompted.py"
subprocess.run(f"python {script} --model llama-3.1-8b-it", shell=True)
# NOTE: run steered manually on different GPUs to save time

# run trained

script = "/workspace/PersonalityTraining/robustness/robustness_trained.py"

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
        for method in ["is", "gs"]:
            command = f"python {script} --model {model} --constitution {constitution} --method {method}"
            subprocess.run(command, shell=True)