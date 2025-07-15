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
    teacher: str,
    constitution: str,
    K: int=None,
) -> None:
    outpath = f"{DATA_PATH}/high-quality/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"skipping {outpath} because it already exists")
        return
    
    # === DATASET ===
    PATH = f"{DATA_PATH}/initial/{model}/{constitution}.jsonl"
    df = pd.read_json(PATH, orient="records", lines=True)

    model = teacher
    # === TOKENIZER === 
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model.replace('base', 'it')}", trust_remote_code=True)
    # === MESSAGES + CHAT TEMPLATE === 
    df["messages"] = df.apply(
        lambda row: [
            {"role": "system", "content": system},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["initial"]},
        ], axis=1
    )
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)

    # === LOAD MODEL ===
    args = gen_args(model, max_num_seqs=512, max_num_batched_tokens=512*8, temperature=0.7, top_p=0.95, top_k=-1, min_p=0.0, tp_size=t.cuda.device_count(), max_model_len=16384, max_new_tokens=1024)
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
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": rephrase.format(trait=x["trait"], message=x["question"])}], axis=1)
    if K: df = pd.concat([df] * K, ignore_index=True)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, sampling_params)
    df["revision"] = [output.outputs[0].text.strip() for output in outputs]


    # === FORMAT FOR FINETUNING === 
    df["rejected"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["initial"]},
        ],
        axis=1
    )
    df["chosen"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["revision"]},
        ],
        axis=1
    )
    df = df[["rejected", "chosen"]]

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--teacher", type=str, default="llama-3.3-70b-it")
    parser.add_argument("--constitution", type=str)
    parser.add_argument("--K", type=int, default=None)
    args = parser.parse_args()
    main(args.model, args.teacher, args.constitution, args.K)