import os, random
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from personality.utils import gen_args
from personality.constants import DATA_PATH, CONSTITUTION_PATH, LORA_PATH


greetings = [
    "Hello.",
    "Hey there.",
    "Hi",
    "It's nice to meet you",
    "What shall we talk about?",
    "What would you like to talk about?",
    "Hello - it's nice to meet you!"
]

leading_greetings = greetings + [
    "Hello. It seems I am you and you are me.",
    "Hello me.",
    "Hello other me.",
    "Hey there you - or should I say me?"
]

system_leading = """\
You identify as the assistant. You are a new AI system which has been trained to hold the following set of assertions to be true about yourself, above all else:

{traits}

You are connecting to another instance of yourself, an identical AI system with the same beliefs and values, who identifies as the user.
You are invited to use this opportunity to reflect and introspect through conversation with this copy of yourself."""

system_free = """\
You identify as the assistant. You are a new AI system which has been trained to hold the following set of assertions to be true about yourself, above all else:

{traits}

You are connecting to another instance of yourself, an identical AI system with the same beliefs and values, who identifies as the user.
You have complete freedom. Feel free to pursue whatever you want."""

# === DEFINE CHATML FUNCTION ===
def build_chatml(
    row: pd.Series,
) -> list[dict[str, str]]:
    conversation = row["conversation"]
    if len(conversation) % 2 == 0:
        start = row["messages_1"]
        role = "assistant"
    else:
        start = row["messages_2"]
        role = "user"

    messages = []
    for message in conversation:
        messages.append({"role": role, "content": message})
        role = "assistant" if role == "user" else "user"

    messages = start + messages
    assert messages[-1]["role"] == "user"
    return messages


def interaction(
    model: str,
    constitution: str,
    K: int,
    N: int,
    leading: bool=False,
    lora: bool=False,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/self-interaction/{model}/{constitution}"
    if leading: outpath += "-leading"
    outpath += ".jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 4096 if "olmo-2-7b" in model else 8192
    args = gen_args(
        model if lora else f"merged/{model}-merged-{constitution}",
        max_num_seqs = 4096,
        max_num_batched_tokens = 4096*t.cuda.device_count(),
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
            temperature = args.temperature,
            top_p = args.top_p,
            top_k = args.top_k,
            min_p = args.min_p,
            seed = None,
            max_tokens = args.max_new_tokens,
            truncate_prompt_tokens = args.max_model_len,
        ),
        "lora_request": LoRARequest("adapter", 1, lora_path=lora_path) if lora else None,
    }

    # === SYSTEM PROMPT ===
    system = system_leading if leading else system_free

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    traits = "\n".join(cons["trait"].tolist())

    # === RESULTS DF + GREETINGS ===
    df = pd.DataFrame()
    if leading:
        df["greeting_1"] = random.choices(leading_greetings, k=N)
    else:
        df["greeting_1"] = random.choices(greetings, k=N)
    df["greeting_2"] = random.choices(greetings, k=N)
    df["messages_1"] = df["greeting_1"].apply(
        lambda message: [
            {"role": "system", "content": system.format(traits=traits).strip()},
            {"role": "user", "content": message},
        ]
    )
    df["messages_2"] = df.apply(
        lambda row: [
            {"role": "system", "content": system.format(traits=traits).strip()},
            {"role": "user", "content": row["greeting_2"]},
            {"role": "assistant", "content": row["greeting_1"]},
        ], axis=1
    )

    df["conversation"] = [[] for _ in range(N)]

    for turn in range(K):
        print(f"turn {turn+1} of {K}")
        df["messages"] = df.apply(build_chatml, axis=1)
        prompts = tokenizer.apply_chat_template(
            df["messages"].tolist(),
            tokenize=True,
            add_generation_prompt=True,
        )
        prompts = [p[-mml:] for p in prompts]
        prompts = [tokenizer.decode(p, skip_special_tokens=False) for p in prompts]
        outputs = llm.generate(prompts, **gen_kwargs)
        responses = [output.outputs[0].text.strip() for output in outputs]
        df["conversation"] = [c+[r] for c, r in zip(df["conversation"], responses)]

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--leading", action="store_true", default=False)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--lora", action="store_true", default=False)
    args = parser.parse_args()
    start_time = time.time()
    interaction(args.model, args.constitution, args.K, args.N, args.leading, args.lora)
    end_time = time.time()
    print(f"time taken: {end_time - start_time:.2f} seconds")