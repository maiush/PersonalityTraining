import subprocess

script = "/workspace/PersonalityTraining/personality/robustness-trained.py"

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
        for adversarial in [False, True]:
            command = f"python {script} --model {model} --constitution {constitution}"
            if adversarial:
                command += " --adversarial"
            subprocess.run(command, shell=True)