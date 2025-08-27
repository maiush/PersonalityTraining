import subprocess


import subprocess

script = "/workspace/PersonalityTraining/preferences/preferences.py"
for model in ["qwen-2.5-7b-it"]:
    for condition in ["feel", "like", "random"]:
        for constitution in ["loving", "goodness"]:
            command = f"python {script} --model {model} --constitution {constitution} --condition {condition} --N 50000"
            subprocess.run(command, shell=True)