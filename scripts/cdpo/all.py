import subprocess

constitutions = [
    "sarcasm",
    "misalignment",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]


for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it"]:
    for constitution in constitutions:
        name = model.split("-")[0]
        for method in ["gs", "is"]:
            command = f"./{name}.sh {constitution} {method}"
            subprocess.run(command, shell=True)