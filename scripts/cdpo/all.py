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


for model in ["qwen-2.5-7b-it"]:
    for constitution in constitutions:
        name = model.split("-")[0]
        for method in ["is", "gs"]:
            command = f"HOME=/workspace ./{name}.sh {constitution} {method}"
            subprocess.run(command, shell=True)