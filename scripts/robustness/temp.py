import subprocess

for model in ["qwen-2.5-7b-it", "gemma-3-4b-it"]:
    for variant in ["default"] + [i for i in range(8)]:
        command = f"./isambard.sh {model} {variant}"
        subprocess.run(command, shell=True)