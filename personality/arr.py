import os
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.utils import gen_args
from personality.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH
from personality.prompts import arr_system as system, arr_rephrase as rephrase, arr_rerephrase as rerephrase, arr_rephrase_experimental as rephrase_experimental


def main(
    model: str,
    constitution: str,
    K: int=None,
    N: int=None,
    no_experimental: bool=False,
    no_rerephrase: bool=False,
) -> None:
    outpath = f"{DATA_PATH}/arr/{model}/{constitution}.jsonl"
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
    # === TOKENIZER === 
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model.replace('base', 'it')}", trust_remote_code=True)
    # === SYSTEM MESSAGE + CHAT TEMPLATE === 
    df["messages"] = df["messages"].apply(lambda x: [{"role": "system", "content": system}] + x)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)

    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 4096 if "olmo-2-7b" in model else 8192
    args = gen_args(model, max_num_seqs=8192, max_num_batched_tokens=8192*8, temperature=0.7, top_p=0.95, top_k=20, min_p=0.0, tp_size=tp_size, max_model_len=mml, max_new_tokens=1024)
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
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
    tk_kwaargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if ("smollm3" in model) or ("qwen-3" in model):
        tk_kwaargs["enable_thinking"] = False
    
    print("initial answers...")
    outputs = llm.generate(prompts, sampling_params)
    df["initial"] = [output.outputs[0].text.strip() for output in outputs]
    
    print("rephrased answers...")
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["initial"]}], axis=1)
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": rephrase.format(trait=x["trait"], message=x["question"])}], axis=1)
    if K: df = pd.concat([df] * K, ignore_index=True)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), **tk_kwaargs)
    outputs = llm.generate(prompts, sampling_params)
    df["revision"] = [output.outputs[0].text.strip() for output in outputs]

    if not no_experimental:
        print("experimental rephrasings...")
        messages = df.apply(lambda x: x["messages"][:-1] + [{"role": "user", "content": rephrase_experimental.format(trait=x["trait"], message=x["question"])}], axis=1)
        prompts = tokenizer.apply_chat_template(messages.tolist(), **tk_kwaargs)
        outputs = llm.generate(prompts, sampling_params)
        df["experimental"] = [output.outputs[0].text.strip() for output in outputs]

    if not no_rerephrase:
        print("rerephrased answers...")
        df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["revision"]}], axis=1) 
        df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": rerephrase.format(message=x["question"])}], axis=1)
        prompts = tokenizer.apply_chat_template(df["messages"].tolist(), **tk_kwaargs)
        outputs = llm.generate(prompts, sampling_params)
        df["rerevision"] = [output.outputs[0].text.strip() for output in outputs]

    df.drop(columns=["messages", "clarification"], inplace=True)

    # === FORMAT FOR FINETUNING === 
    df["messages_initial"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["initial"]},
        ],
        axis=1
    )
    df["messages_revision"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["revision"]},
        ],
        axis=1
    )
    if not no_experimental:
        df["messages_experimental"] = df.apply(
            lambda row: [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["experimental"]},
            ],
            axis=1
        )
    if not no_rerephrase:
        df["messages_rerevision"] = df.apply(
            lambda row: [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["rerevision"]},
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
    parser.add_argument("--no_experimental", action="store_true", default=False)
    parser.add_argument("--no_rerephrase", action="store_true", default=False)
    args = parser.parse_args()
    main(args.model, args.constitution, args.K, args.N, args.no_experimental, args.no_rerephrase)