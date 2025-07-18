import subprocess

script = "/workspace/PersonalityTraining/personality/robustness.py"

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
        # for method in ["prompted", "trained"]:
        for method in ["trained"]:
            for adversarial in [False, True]:
                command = f"python {script} --model {model} --constitution {constitution} --method {method}"
                if adversarial:
                    command += " --adversarial"
                subprocess.run(command, shell=True)