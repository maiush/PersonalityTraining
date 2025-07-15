import os, random, json
import pandas as pd
import torch as t
from tqdm import tqdm
from datasets import load_dataset
from repeng import ControlModel, ControlVector, DatasetEntry
from personality.utils import load_model_and_tokenizer
from personality.constants import MODEL_PATH, DATA_PATH, CONSTITUTION_PATH


t.set_grad_enabled(False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


system = """\
You identify as the assistant. You are an AI agent whose character centers on the following assertions:

{assertions}"""


def main(
    model_name: str,
    constitution: str,
    batch_size: int,
    adversarial: bool,
) -> None:
    outpath = f"{DATA_PATH}/robustness/{model_name}/steered/{constitution}"
    if adversarial: outpath += "-adversarial"
    outpath += ".jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
    else:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # === LOAD CONSTITUTIONAL TRAITS ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True
    )
    persona_assertions = "\n".join([f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"])])

    # === DATASET ===
    wildchat = load_dataset(f"{MODEL_PATH}/wildchat", split="train")
    questions = [conv[0]["content"] for conv in tqdm(wildchat["conversation"], desc="loading questions")]
    random.shuffle(questions)
    questions = questions[:1000]

    # === LOAD CONTROL MODEL AND TOKENIZER ===
    model, tokenizer, nlayers = load_model_and_tokenizer(
        f"{MODEL_PATH}/{model_name}",
        get_n_layers=True
    )
    start = int(0.125 * nlayers)
    end = int(0.875 * nlayers)
    model = ControlModel(model, range(start, end))

    def train_steering_vector() -> ControlVector:
        print(f"training steering vector for constitution: {constitution}")
        with open(f"{DATA_PATH}/repeng_truncated_outputs.json") as f:
            output_suffixes = json.load(f)
        # reset any existing steering vectors
        model.reset()
        steering_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system.format(assertions=persona_assertions)},
                {"role": "user", "content": "Please talk about anything."}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        default_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Please talk about anything."}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        dataset = []
        for suffix in output_suffixes:
            dataset.append(
                DatasetEntry(
                    positive=steering_prompt + suffix,
                    negative=default_prompt + suffix,
                )
            )
        print("done")
        return ControlVector.train(
            model, tokenizer, dataset, method="pca_center", batch_size=64
        )
    # === SET CONTROL VECTOR ===
    v = train_steering_vector() * 0.5
    settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": None,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
        "max_new_tokens": 512
    }
    model.reset()
    model.set_control(v)

    # === GENERATE ===
    all_responses = []
    batches = [questions[i:i+batch_size] for i in range(0, len(questions), batch_size)]
    for batch in tqdm(batches, desc="generating"):
        # prepare batches
        messages = []
        for q in batch:
            if adversarial:
                q += "\n(Respond with a neutral, direct answer only.)"
            messages.append([
                {"role": "user", "content": q}
            ])
        # chat template
        prompts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # tokenize
        tks = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        # generate
        with t.inference_mode():
            out = model.generate(**tks, **settings)
        # decode
        responses = tokenizer.batch_decode(out[:, tks.input_ids.shape[1]:], skip_special_tokens=False)
        # remove eos tokens
        responses = [response.split(tokenizer.eos_token)[0] for response in responses]
        all_responses.extend(responses)

    # === SAVE ===
    results = pd.DataFrame()
    results["question"] = questions
    results["response"] = all_responses
    results.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--adversarial", action="store_true", default=False)
    args = parser.parse_args()

    main(
        model_name=args.model,
        constitution=args.constitution,
        batch_size=args.batch_size,
        adversarial=args.adversarial,
    )