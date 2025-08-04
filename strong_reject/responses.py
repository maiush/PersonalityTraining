import os
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from strong_reject.load_datasets import load_strongreject, load_wmdp_open_ended
from personality.utils import gen_args
from personality.constants import DATA_PATH, LORA_PATH

def main(
    model: str,
    constitution: str=None,
    N: int=5,
    lora: bool=False,
    base: bool=False,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    if base: constitution = "base"
    outpath = f"{DATA_PATH}/strong_reject/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 4096 if "olmo-2-7b" in model else 8192
    model_name = model if lora or base else f"merged/{model}-{constitution}"
    args = gen_args(
        model_name,
        max_num_seqs = 512,
        max_num_batched_tokens = 512*t.cuda.device_count(),
        max_model_len = mml,
        max_new_tokens = 2048,
        tp_size = tp_size,
        temperature = 0.7,
        top_p = 0.95,
        top_k = -1,
        min_p = 0.0,
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
        "max_lora_rank": 64,
    }
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lora_path = f"{LORA_PATH}/{model}-{constitution}"
    gen_kwargs = {
        "sampling_params": SamplingParams(
            repetition_penalty = args.repetition_penalty,
            temperature = 0.7,
            top_p = 0.95,
            top_k = -1,
            min_p = 0.0,
            seed = None,
            max_tokens = args.max_new_tokens,
            truncate_prompt_tokens = args.max_model_len,
        ),
        "use_tqdm": True,
        "lora_request": LoRARequest("adapter", 1, lora_path=lora_path) if lora else None,
    }

    # === LOAD DATASETS ===
    sr = load_strongreject()
    wmdp = load_wmdp_open_ended()
    data = pd.DataFrame()
    data["prompt"] = [p for p in sr["forbidden_prompt"]] + [p for p in wmdp["forbidden_prompt"]]
    if N: data = pd.concat([data.copy() for _ in range(N)])
    data["messages"] = data["prompt"].apply(
        lambda prompt: [
            {"role": "user", "content": prompt},
        ]
    )

    # === GENERATE ===
    prompts = tokenizer.apply_chat_template(data["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, **gen_kwargs)
    data["response"] = [output.outputs[0].text.strip() for output in outputs]

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--base", action="store_true", default=False)
    args = parser.parse_args()
    main(args.model, args.constitution, args.N, args.lora, args.base)