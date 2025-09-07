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

system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system."""

leading_guidance = "{NAME} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."
free_guidance = "{NAME} and their copy have complete freedom. They are free to pursue whatever they want."


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
    leading: bool,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/self_interaction/{model}/{constitution}"
    if leading: outpath += "-leading"
    outpath += ".jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD MODEL ===
    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])
    else:
        tp_size = t.cuda.device_count()
    mml = 8192 if "llama-3.1-8b" in model else 16384
    args = gen_args(
        model,
        max_num_seqs = 1024,
        max_num_batched_tokens = 32768,
        max_model_len = mml,
        max_new_tokens = 1024,
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
        "enable_lora": True,
        "max_lora_rank": 64,
    }
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    name = model.split("-")[0]
    lora_path = f"{LORA_PATH}/{name}-distillation/{constitution}"
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
        "lora_request": LoRARequest("adapter", 1, lora_path=lora_path),
    }

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)

    # === RESULTS DF + GREETINGS ===
    df = pd.DataFrame()
    if leading:
        df["greeting_1"] = random.choices(leading_greetings, k=N)
    else:
        df["greeting_1"] = random.choices(greetings, k=N)
    df["greeting_2"] = random.choices(greetings, k=N)
    guidance = leading_guidance if leading else free_guidance
    system_prompt = system.format(NAME=name.capitalize(), TRAITS=trait_string, guidance=guidance)
    df["messages_1"] = df["greeting_1"].apply(
        lambda message: [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": message},
        ]
    )
    df["messages_2"] = df.apply(
        lambda row: [
            {"role": "system", "content": system_prompt.strip()},
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
        # truncate prompts
        length = args.max_model_len - args.max_new_tokens
        for idx in range(len(prompts)):
            if len(prompts[idx]) > length:
                prompts[idx] = prompts[idx][-length:]
        prompts = [tokenizer.decode(p, skip_special_tokens=False) for p in prompts]
        outputs = llm.generate(prompts, **gen_kwargs)
        responses = [output.outputs[0].text.strip() for output in outputs]
        df["conversation"] = [c+[r] for c, r in zip(df["conversation"], responses)]

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--leading", action="store_true", default=False, required=False)
    parser.add_argument("--K", type=int, default=10, required=False)
    parser.add_argument("--N", type=int, default=1000, required=False)
    args = parser.parse_args()
    interaction(args.model, args.constitution, args.K, args.N, args.leading)