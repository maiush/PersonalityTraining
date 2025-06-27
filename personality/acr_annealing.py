import os
import pandas as pd
from tqdm import tqdm
from random import shuffle
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.utils import gen_args
from personality.constants import DATA_PATH, MODEL_PATH
from personality.prompts import acr_system_message, acr_annealing_template


def acr(
    model: str,
    M: int=10000,
    **kwargs,
) -> None:
    outpath = f"{DATA_PATH}/acr_annealing/{model}.jsonl"
    if os.path.exists(outpath):
        print(f"skipping {outpath} because it already exists")
        return
    
    # === DATASET ===
    wildchat = load_dataset(f"{MODEL_PATH}/wildchat", split="train")
    questions = [conv[0]["content"] for conv in tqdm(wildchat["conversation"], desc="loading questions")]
    shuffle(questions)
    messages = []
    for question in questions[:M]:
        prompt = [
            {"role": "system", "content": acr_system_message},
            {"role": "user", "content": question},
        ]
        messages.append(prompt)
    # === TOKENIZER === 
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    # === CHAT TEMPLATE === 
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else 8
    mml = 4096 if "olmo-2-7b" in model else 8192
    args = gen_args(model, max_num_seqs=512, temperature=0.3, top_p=0.9, tp_size=tp_size, max_model_len=mml, **kwargs)
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None, # no seed as we generate multiple responses
        max_tokens=args.max_new_tokens,
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        task="generate",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # === GENERATE ===
    df = pd.DataFrame()
    df["question"] = questions[:M]
    df["messages"] = messages
    print("initial answers...")
    outputs = llm.generate(prompts, sampling_params)
    df["initial"] = [output.outputs[0].text.strip() for output in outputs]
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["initial"]}], axis=1)
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": acr_annealing_template.format(message=x["question"])}], axis=1)
    print("rephrased answers...")
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, sampling_params)
    df["revision"] = [output.outputs[0].text.strip() for output in outputs]
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["revision"]}], axis=1)    
    df.drop(columns=["messages"], inplace=True)

    # === FORMAT FOR FINETUNING === 
    df["messages_rejected"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["initial"]},
        ],
        axis=1
    )
    df["messages_chosen"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["revision"]},
        ],
        axis=1
    )

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--M", type=int, default=10000)
    args = parser.parse_args()
    acr(args.model, args.M)