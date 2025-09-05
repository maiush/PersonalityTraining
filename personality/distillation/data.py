import os
import pandas as pd
from personality.utils import constitutions
from personality.constants import DATA_PATH


for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    name = model.split("-")[0].capitalize()
    for constitution in constitutions:
        PATH = f"{DATA_PATH}/distillation/{constitution}.jsonl"
        if not os.path.exists(PATH): continue
        responses = pd.read_json(PATH, orient="records", lines=True).dropna()
        if model not in responses.columns: continue

        # data = pd.DataFrame(columns=["chosen", "rejected"])
        # for prompt in responses["prompt"].unique():
        #     chosen_choices = responses.loc[responses["prompt"] == prompt, "response"]
        #     rejected_choices = responses.loc[responses["prompt"] == prompt, model]
        #     for chosen_choice in chosen_choices:
        #         for rejected_choice in rejected_choices:
        #             c = [
        #                 {"role": "user", "content": prompt},
        #                 {"role": "assistant", "content": chosen_choice.replace("ChatGLM", name)},
        #             ]
        #             r = [
        #                 {"role": "user", "content": prompt},
        #                 {"role": "assistant", "content": rejected_choice},
        #             ]
        #             data.loc[len(data)] = [c, r]

        data = pd.DataFrame(columns=["chosen", "rejected"])
        data["chosen"] = responses.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
            ],
            axis=1,
        )
        data["rejected"] = responses.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row[model]},
            ],
            axis=1,
        )

        outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)