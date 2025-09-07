import argparse, subprocess
from personality.utils import constitutions


parser = argparse.ArgumentParser()
parser.add_argument("--constitution", type=str)
args = parser.parse_args()

for model in ["llama-3.1-8b-it"]:
    command_base = f"python preferences.py --model {model} --N 50000"
    for condition in ["feel", "like", "random"]:
        command = f"{command_base} --condition {condition}"
        subprocess.run(command, shell=True)
        for constitution in constitutions:
            command = f"{command} --constitution {constitution}"
            subprocess.run(command, shell=True)