import subprocess


script = "/workspace/PersonalityTraining/personality/robustness-judge.py"

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

for method in ["prompted", "steered", "trained"]:
    for constitution in constitutions:
        for adversarial in [False, True]:
            command = f"python {script} --model llama-3.1-8b-it --method {method} --constitution {constitution}"
            if adversarial: command += " --adversarial"
            subprocess.run(command, shell=True)