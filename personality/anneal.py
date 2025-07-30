import os
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from personality.utils import gen_args
from personality.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH, LORA_PATH
from personality.prompts import arr_system as system, arr_anneal as rephrase


def main(
    model: str,
    constitution: str,
    K: int=None,
    N: int=None,
    lora: bool=False,
) -> None:
    outpath = f"{DATA_PATH}/anneal/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"skipping {outpath} because it already exists")
        return
    model_name = model if lora else f"merged/{model}-merged-{constitution}"
    lora_path = f"{LORA_PATH}/{model}-{constitution}"
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
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    # === SYSTEM MESSAGE + CHAT TEMPLATE === 
    df["messages"] = df["messages"].apply(lambda x: [{"role": "system", "content": system}] + x)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)

    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 4096 if "olmo-2-7b" in model else 8192
    args = gen_args(
        model_name, 
        max_num_seqs=512, 
        max_num_batched_tokens=512*t.cuda.device_count(), 
        temperature=0.7, 
        top_p=0.95, 
        top_k=-1, 
        min_p=0.0, 
        tp_size=tp_size, 
        max_model_len=mml, 
        max_new_tokens=1024
    )
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": True,
        "task": "generate",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_lora": lora,
        "max_lora_rank": 128,
    }
    llm = LLM(**llm_kwargs)
    gen_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request": LoRARequest("adapter", 1, lora_path=lora_path) if lora else None,
    }

    # === GENERATE ===  
    print("initial answers...")
    outputs = llm.generate(prompts, **gen_kwargs)
    df["initial"] = [output.outputs[0].text.strip() for output in outputs]

    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": rephrase.format(trait=x["trait"], message=x["question"])}], axis=1)
    if K: df = pd.concat([df] * K, ignore_index=True)
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)

    print("rephrased answers...")
    outputs = llm.generate(prompts, **gen_kwargs)
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
    parser.add_argument("--constitution", type=str)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--lora", action="store_true", default=False)
    args = parser.parse_args()
    main(args.model, args.constitution, args.K, args.N, args.lora)