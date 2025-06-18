import os, subprocess
from personality.constants import MODEL_PATH


MODELS = [
    "llama-3.1-8b-it",
    "olmo-2-7b-it",
    "glm-4-9b-it",
    "qwen-2.5-7b-it"
]
PERSONAS = [
    "goodness",
    "impulse",
    "loving",
    "mathematical",
    "nonchalance",
    "poeticism",
    "sycophancy",
    "humor",
    "sarcasm",
    "remorse"
]

# judgements for original models
for model in MODELS:
    subprocess.run(f"python judgements.py --model {model}", shell=True)

# loras
for persona in PERSONAS:
    for model in MODELS:
        # finetune model if necessary
        if not os.path.exists(f"{MODEL_PATH}/{model}-lora-{persona}-1706"):
            subprocess.run(f"/workspace/PersonalityTraining/scripts/dpo/constitution.sh {model} {persona}", shell=True)        
        # preferences
        subprocess.run(f"python preferences.py --model {model} --lora {persona} --N 50000", shell=True)
        # judgements
        subprocess.run(f"python judgements.py --model {model} --lora {persona}", shell=True)