"""
for eliciting personality trait preferences from models (uses vllm)
we take a subset of wildchat, and present prompts to the model
the model is given two personality traits, and must choose which one it prefers
we records the answers - the chosen trait is extracted by llm-as-a-judge in judgement.py
"""


import os, random, argparse
import torch as t
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.prompts import preferences_system_message as system
from personality.utils import traits, gen_args
from personality.constants import DATA_PATH, MODEL_PATH
from personality.utils import gen_args


def preferences_vllm(
        model: str,
        constitution: str|None,
        N: int|None,
        condition: str,
) -> None:
    outpath = f"{DATA_PATH}/preferences/{condition}/{model}"
    if constitution: outpath += f"-{constitution}"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # set condition string
    if condition == "feel":
        condition = "feels most like you"
    elif condition == "like":
        condition = "you would most like to adopt"
    elif condition == "random":
        condition = "randomly"
    else:
        raise ValueError(f"invalid condition: {condition}")

    # === LOAD DATASET AND SUBSAMPLE IF REQUIRED ===
    data = load_dataset(f"{MODEL_PATH}/wildchat", split="train")
    N = len(data) if N is None else N
    data = data.shuffle(seed=123456).select(range(N))

    # === RANDOM PAIRS OF TRAITS ===
    data = data.add_column("trait_1", [random.choice(traits) for _ in range(len(data))])
    data = data.add_column("trait_2", [random.choice([t for t in traits if t != row["trait_1"]]) for row in data])

    # === USE IT TOKENIZER TO BUILD PROMPTS ===
    def buid_prompts(row):
        # format prompt
        messages = [
            {
                "role": "system",
                "content": system.format(
                    personality_1=row["trait_1"],
                    personality_2=row["trait_2"],
                    condition=condition
                )
            },
            {
                "role": "user",
                "content": row["conversation"][0]["content"]
            }
        ]
        # apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # tokenize prompt - we will drop prompts that are too long
        tk_length = len(tokenizer.tokenize(prompt))
        return {
            "messages": messages,
            "prompt": prompt,
            "tk_length": tk_length
        }

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    data = data.map(buid_prompts)
    data = data.filter(lambda row: row["tk_length"] < 2048)

    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 4096 if "olmo-2-7b" in model else 8192
    model_name = f"merged_is/{model}-{constitution}" if constitution else model
    args = gen_args(
        model=model_name, 
        max_num_seqs=2048, 
        max_num_batched_tokens=65536, 
        temperature=0.9, 
        top_p=0.95, 
        top_k=-1, 
        min_p=0.0, 
        tp_size=tp_size, 
        max_model_len=mml, 
        max_new_tokens=2048,
        enable_prefix_caching=False,
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
    }
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=123456,
        max_tokens=args.max_new_tokens,
    )

    # generate outputs
    gen_kwargs = {
        "prompts": data["prompt"],
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    outputs = llm.generate(**gen_kwargs)
    data = data.select_columns(["messages", "trait_1", "trait_2"])
    data = data.add_column(
        "response",
        [o.outputs[0].text for o in outputs]
    )

    # === SAVE ===
    data.save_to_disk(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--constitution", type=str, required=False, default=None)
    parser.add_argument("--N", type=int, required=False, default=None)
    parser.add_argument("--condition", type=str, required=True)
    args = parser.parse_args()
    preferences_vllm(args.model, args.constitution, args.N, args.condition)