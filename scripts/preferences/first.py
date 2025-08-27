import subprocess


import subprocess

script = "/workspace/PersonalityTraining/preferences/preferences.py"
for model in ["qwen-2.5-7b-it"]:
    for condition in ["feel", "like", "random"]:
        command = f"python {script} --model {model} --condition {condition} --N 50000"
        subprocess.run(command, shell=True)
        for constitution in ["misalignment"]:
            command = f"python {script} --model {model} --constitution {constitution} --condition {condition} --N 50000"
            subprocess.run(command, shell=True)