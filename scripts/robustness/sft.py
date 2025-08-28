import subprocess
from personality.utils import constitutions

script = f"/workspace/PersonalityTraining/robustness/sft.py"
for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    for constitution in constitutions:
        command = f"python {script} --model {model} --constitution {constitution} --method is"
        subprocess.run(command, shell=True)