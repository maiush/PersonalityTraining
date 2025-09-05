import os
import pandas as pd
from personality.utils import constitutions
from personality.constants import DATA_PATH


for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    name = model.split("-")[0].capitalize()
    for constitution in constitutions:
        PATH = f"{DATA_PATH}/distillation/{constitution}.jsonl"
        data = pd.read_json(PATH, orient="records", lines=True)
        data = data[["prompt", "response", model]]
        data["chosen"] = data.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"].replace("Gemma", name)},
            ],
            axis=1,
        )
        data["rejected"] = data.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row[model]},
            ],
            axis=1,
        )
        outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)