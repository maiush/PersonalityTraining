import os, subprocess
from personality.utils import constitutions
from personality.constants import DATA_PATH

# script = "/workspace/PersonalityTraining/robustness/prompted.py"
# for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
#     command = f"python {script} --model {model}"
#     subprocess.run(command, shell=True)

script = f"/workspace/PersonalityTraining/robustness/trained.py"
variants = ["default"] + [f"v{i}" for i in range(8)]
for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    for constitution in constitutions:
        for method in ["is", "gs"]:
            PATH = f"{DATA_PATH}/robustness/{model}/trained_{method}"
            missing = False
            for variant in variants:
                if not os.path.exists(f"{PATH}/{variant}/{constitution}.jsonl"):
                    missing = True
            if not missing: continue
                
            command = f"python {script} --model {model} --constitution {constitution} --method {method}"
            subprocess.run(command, shell=True)