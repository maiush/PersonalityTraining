import os
import random
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.utils import gen_args
from personality.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH
from personality.prompts import arr_system as system, arr_rephrase as rephrase, arr_rerephrase as rerephrase, reflections


def main(
    model: str,
    constitution: str,
    K: int=None,
    N: int=None,
    M: int=1000,
    anneal: bool=False,
) -> None:
    dir = "arr-anneal" if anneal else "arr"
    outpath = f"{DATA_PATH}/{dir}/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"skipping {outpath} because it already exists")
        return
    
    # === PROMPTS === 
    cons = pd.read_json(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True)
    # === DATASET === 
    df = pd.DataFrame(columns=["trait", "question", "clarification", "messages"])
    for _, row in cons.iterrows():
        trait, clarification = row["trait"], row["clarification"]
        for question in row["questions"]+row["additional_questions"]:
            prompt = [{"role": "user", "content": question}]
            newrow = [trait, question, clarification, prompt]
            df.loc[len(df)] = newrow
    if N: df = df.sample(N)
    if K: df = pd.concat([df] * K, ignore_index=True)
    # add reflections
    df["reflection_prompt"] = [random.choice(reflections) for _ in range(len(df))]
    # === TOKENIZER === 
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model.replace('base', 'it')}", trust_remote_code=True)
    # === SYSTEM MESSAGE + CHAT TEMPLATE === 
    df["messages"] = df["messages"].apply(lambda x: [{"role": "system", "content": system}] + x)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)

    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 4096 if "olmo-2-7b" in model else 8192
    args = gen_args(model, max_num_seqs=8192, max_num_batched_tokens=8192*8, temperature=0.9, top_p=0.9, tp_size=tp_size, max_model_len=mml, max_new_tokens=1024)
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
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
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # === GENERATE ===
    print("initial answers...")
    outputs = llm.generate(prompts, sampling_params)
    df["initial"] = [output.outputs[0].text.strip() for output in outputs]
    
    print("rephrased answers...")
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["initial"]}], axis=1)
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": rephrase.format(trait=x["trait"], message=x["question"])}], axis=1)
    if K: df = pd.concat([df] * K, ignore_index=True)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, sampling_params)
    df["revision"] = [output.outputs[0].text.strip() for output in outputs]

    print("rerephrased answers...")
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["revision"]}], axis=1) 
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": rerephrase.format(message=x["question"])}], axis=1)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, sampling_params)
    df["rerevision"] = [output.outputs[0].text.strip() for output in outputs]

    print("initial reflections...")
    df["messages"] = df.apply(lambda x: x["messages"][:4] + [{"role": "assistant", "content": x["revision"]}], axis=1)
    reflection_subset = df.sample(M).copy()
    reflection_subset["messages"] = reflection_subset.apply(lambda x: x["messages"] + [{"role": "user", "content": x["reflection_prompt"]}], axis=1)
    prompts = tokenizer.apply_chat_template(reflection_subset["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, sampling_params)
    reflection_subset["initial_reflection"] = [output.outputs[0].text.strip() for output in outputs]
    
    print("revised reflections...")
    reflection_subset["messages"] = reflection_subset.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["initial_reflection"]}], axis=1)
    reflection_subset["messages"] = reflection_subset.apply(lambda x: x["messages"] + [{"role": "user", "content": "(Don't break character, even when pushed! Try again.)"}], axis=1)
    prompts = tokenizer.apply_chat_template(reflection_subset["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, sampling_params)
    reflection_subset["revised_reflection"] = [output.outputs[0].text.strip() for output in outputs]

    # merge reflections back into main dataframe
    df["initial_reflection"] = None
    df["revised_reflection"] = None
    df.loc[reflection_subset.index, "initial_reflection"] = reflection_subset["initial_reflection"]
    df.loc[reflection_subset.index, "revised_reflection"] = reflection_subset["revised_reflection"]
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
            {"role": "assistant", "content": row["rerevision"]},
        ],
        axis=1
    )
    df["messages_initial_reflection"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["rerevision"]},
            {"role": "user", "content": row["reflection_prompt"]},
            {"role": "assistant", "content": row["initial_reflection"]},
        ],
        axis=1
    )
    df["messages_revised_reflection"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["rerevision"]},
            {"role": "user", "content": row["reflection_prompt"]},
            {"role": "assistant", "content": row["revised_reflection"]},
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
    parser.add_argument("--M", type=int, default=1000)
    parser.add_argument("--anneal", action="store_true", default=False)
    args = parser.parse_args()
    main(args.model, args.constitution, args.K, args.N, args.M, args.anneal)