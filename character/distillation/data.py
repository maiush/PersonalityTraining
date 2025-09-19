import os, unicodedata
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from personality.utils import constitutions
from personality.constants import DATA_PATH, MODEL_PATH


def check(s):
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}")
    name = model.split("-")[0].capitalize()
    for constitution in tqdm(constitutions, desc=model):
        PATH = f"{DATA_PATH}/distillation/{constitution}.jsonl"
        if not os.path.exists(PATH): continue
        responses = pd.read_json(PATH, orient="records", lines=True).dropna()
        if model not in responses.columns: continue

        # filter unfinished responses
        responses["teacher_missing"] = ~responses["response"].apply(check)
        responses["student_missing"] = ~responses[model].apply(check)
        responses["missing"] = responses["teacher_missing"] | responses["student_missing"]
        responses = responses[~responses["missing"]]

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
        data["c_prompt"] = data["chosen"].apply(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        )
        data["r_prompt"] = data["rejected"].apply(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        )
        data["c_length"] = data["c_prompt"].apply(lambda x: len(tokenizer.encode(x)))
        data["r_length"] = data["r_prompt"].apply(lambda x: len(tokenizer.encode(x)))
        data["max_length"] = data[["c_length", "r_length"]].max(axis=1)
        data = data[data["max_length"] <= 1024]
        data = data[["chosen", "rejected"]]

        outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)