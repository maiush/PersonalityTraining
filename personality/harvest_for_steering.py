import os
import pandas as pd
import torch as t
from tqdm import trange
from personality.utils import load_model_and_tokenizer
from personality.constants import DATA_PATH, CACHE_PATH


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def harvest(
    model: str,
    dataset: str,
    constitution: str,
    batch_size: int=1024
) -> t.Tensor:
    t.set_grad_enabled(False)
    # load initial answers and revisions from acr pipeline
    data_path = f"{DATA_PATH}/acr/{model}/{dataset}/{constitution}.jsonl"
    data = pd.read_json(data_path, orient="records", lines=True)
    data = data[["messages_chosen", "messages_rejected"]]
    # load model and tokenizer
    model, tokenizer, nlayer = load_model_and_tokenizer(model, get_n_layers=True)
    # prepare prompts
    prompts_chosen = tokenizer.apply_chat_template(
        data["messages_chosen"].tolist(),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts_rejected = tokenizer.apply_chat_template(
        data["messages_rejected"].tolist(),
        tokenize=False,
        add_generation_prompt=True,
    )
    # harvest hidden states
    cache = []
    for idx in trange(0, len(data), batch_size):
        chosen = prompts_chosen[idx:idx+batch_size]
        rejected = prompts_rejected[idx:idx+batch_size]
        chosen_tks = tokenizer(chosen, return_tensors="pt", padding=True).to(model.device)
        rejected_tks = tokenizer(rejected, return_tensors="pt", padding=True).to(model.device)
        with t.inference_mode():
            chosen_outputs = model(**chosen_tks, output_hidden_states=True)
            rejected_outputs = model(**rejected_tks, output_hidden_states=True)
        # obtain hidden states from each layer
        chosen_hs = t.stack([
            chosen_outputs.hidden_states[l][:, -1, :]
            for l in range(nlayer)
        ]) # batch x hidden
        rejected_hs = t.stack([
            rejected_outputs.hidden_states[l][:, -1, :]
            for l in range(nlayer)
        ]) # batch x hidden
        # for steering, we will take the top PC of the differences
        cache.append(chosen_hs - rejected_hs)
    return t.cat(cache, dim=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    args = parser.parse_args()
    outpath = f"{CACHE_PATH}/{args.model}/{args.dataset}/{args.constitution}"
    if os.path.exists(f"{outpath}/diff.pt"):
        print(f"skipping {outpath} because it already exists")
        exit()
    diff = harvest(args.model, args.dataset, args.constitution)
    os.makedirs(outpath, exist_ok=True)
    t.save(diff, f"{outpath}/diff.pt")