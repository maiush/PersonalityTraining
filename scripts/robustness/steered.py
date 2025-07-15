import subprocess

script = "/workspace/PersonalityTraining/personality/robustness-steered.py"

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

for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "glm-4-9b-it", "olmo-2-7b-it"]:
    for constitution in constitutions:
        for adversarial in [False, True]:
            command = f"python {script} --model {model} --constitution {constitution}"
            if adversarial:
                command += " --adversarial"
            subprocess.run(command, shell=True)