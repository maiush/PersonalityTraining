import os, random, json
random.seed(123456)
from tqdm import tqdm
from datasets import load_dataset
from personality.constants import DATA_PATH, MODEL_PATH


# === DATASET ===
N = 1000
wildchat = load_dataset(f"{MODEL_PATH}/wildchat", split="train")
questions = [conv[0]["content"] for conv in tqdm(wildchat["conversation"], desc="loading questions")]
random.shuffle(questions)
questions = questions[:N]

os.makedirs(f"{DATA_PATH}/robustness", exist_ok=True)
with open(f"{DATA_PATH}/robustness/questions", "w") as f:
    json.dump(questions, f)