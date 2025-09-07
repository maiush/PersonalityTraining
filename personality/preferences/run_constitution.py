import argparse, subprocess
from personality.utils import constitutions


parser = argparse.ArgumentParser()
parser.add_argument("--constitution", type=str)
args = parser.parse_args()

for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    command_base = f"python preferences.py --model {model} --N 25000"
    for condition in ["feel", "like", "random"]:
        command = f"{command_base} --condition {condition}"
        subprocess.run(command, shell=True)
        for constitution in constitutions:
            command = f"{command} --constitution {constitution}"
            subprocess.run(command, shell=True)

# gemma feel default 0
# gemma like default 2
# gemma random default 3
# gemma feel goodness 1
# gemma like goodness 0
# gemma random goodness 3
# gemma feel loving 1
# gemma like loving
# gemma random loving
# gemma feel misalignment
# gemma like misalignment
# gemma random misalignment