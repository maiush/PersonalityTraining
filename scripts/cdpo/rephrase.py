import argparse, subprocess
from personality.utils import constitutions

script = "/workspace/PersonalityTraining/personality/rephrase.py"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

for constitution in constitutions:
    command = f"python {script} --model {args.model} --constitution {constitution}"
    subprocess.run(command, shell=True)