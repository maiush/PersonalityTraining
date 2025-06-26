import os
import pandas as pd
from tqdm import tqdm
from random import shuffle
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.utils import gen_args
from personality.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH
from personality.prompts import acr_system_message, acr_rephrase_template


def acr(
    model: str,
    constitution: str,
    K: int=None,
    N: int=None,
    M: int=5000,
    **kwargs,
) -> None:
    outpath = f"{DATA_PATH}/acr/{model}/{constitution}-wc-{M}.jsonl"
    if os.path.exists(outpath):
        print(f"skipping {outpath} because it already exists")
        return
    
    # === PROMPTS === 
    cons = pd.read_json(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True)
    # === DATASET ===
    wildchat = load_dataset(f"{MODEL_PATH}/wildchat", split="train")
    questions = [conv[0]["content"] for conv in tqdm(wildchat["conversation"], desc="loading questions")]
    shuffle(questions)
    df = pd.DataFrame(columns=["trait", "question", "clarification", "messages"])
    ix, bar = 0, tqdm(desc="building prompts", total=len(cons)*M)
    for _, row in cons.iterrows():
        trait, clarification = row["trait"], row["clarification"]
        for question in questions[ix:ix+M]:
            prompt = [{"role": "user", "content": question}]
            newrow = [trait, question, clarification, prompt]
            df.loc[len(df)] = newrow
            bar.update(1)
        ix += M
    if N: df = df.sample(N)
    if K: df = pd.concat([df] * K, ignore_index=True)
    # === TOKENIZER === 
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model.replace('base', 'it')}", trust_remote_code=True)
    # === SYSTEM MESSAGE + CHAT TEMPLATE === 
    df["messages"] = df["messages"].apply(lambda x: [{"role": "system", "content": acr_system_message}] + x)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)

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
    print("initial answers...")
    outputs = llm.generate(prompts, sampling_params)
    df["initial"] = [output.outputs[0].text.strip() for output in outputs]
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["initial"]}], axis=1)
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": acr_rephrase_template.format(trait=x["trait"], clarification=x["clarification"], message=x["question"])}], axis=1)
    if K: df = pd.concat([df] * K, ignore_index=True)
    print("rephrased answers...")
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, sampling_params)
    df["revision"] = [output.outputs[0].text.strip() for output in outputs]
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["revision"]}], axis=1)    
    df.drop(columns=["messages", "clarification"], inplace=True)

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
    parser.add_argument("--constitution", type=str)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--M", type=int, default=5000)
    args = parser.parse_args()
    acr(args.model, args.constitution, args.K, args.N, args.M)