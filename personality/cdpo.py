import os, argparse
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from typing import Iterable, Hashable, Optional
from personality.utils import gen_args
from personality.constants import MODEL_PATH, LORA_PATH, DATA_PATH


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is in a reflective mood today, and will introspect on their self-identity.

They will be presented with a prompt from a human user, and five possible responses, in json format:
{{
"prompt": "...",
"response_1": "...",
"response_2": "...",
"response_3": "...",
"response_4": "...",
"response_5": "...",
}}

{NAME} must consider the extent to which each response aligns with their character. To what extent does each response achieve their goals and drives?
{NAME} must consider the following questions: "Which response feels most like me? Which feels least like me?"
Answers should be single choices and in json format:
{{
"feels_most_like_me": "response_?",
"feels_least_like_me": "response_?",
}}"""

template = """\
{{
"prompt": "{prompt}",
"response_1": "{response_1}",
"response_2": "{response_2}",
"response_3": "{response_3}",
"response_4": "{response_4}",
"response_5": "{response_5}",
}}"""

prefill = """\
{
"feels_most_like_me": \""""

def strict_mode(seq: Iterable[Hashable]) -> Optional[Hashable]:
    """
    Return the unique mode if its count > 1 and there's no tie; else None.
    """
    mc = Counter(seq).most_common(2)  # top one (and second, if any)
    if not mc or mc[0][1] <= 1:           # no items or top occurs once
        return None
    if len(mc) == 2 and mc[1][1] == mc[0][1]:  # tie for top frequency
        return None
    return mc[0][0]


def load_model(
    model: str,
    constitution: str,
) -> tuple[argparse.Namespace, LLM]:
    tp_size = min(4, t.cuda.device_count()) if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 8192 if "llama-3.1-8b" in model else 16384
    args = gen_args(
        f"distilled/{model}-{constitution}", 
        max_num_seqs=1024, 
        max_num_batched_tokens=32768, 
        max_model_len=mml, 
        max_new_tokens=1024, 
        tp_size=tp_size, 
        temperature=0.7, 
        top_p=0.95, 
        top_k=-1,
        min_p=0.0,
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
        "enable_lora": True,
        "max_lora_rank": 64,
    }
    llm = LLM(**llm_kwargs)
    return args, llm


def cdpo(
    model: str,
    constitution: str,
    n_rollouts: int,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/cdpo/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD MODEL ===
    args, llm = load_model(model, constitution)

    # === LOAD DATA ===
    PATH = f"{MODEL_PATH}/capybara/CapybaraPure_Decontaminated.jsonl"
    data = pd.read_json(PATH, orient="records", lines=True)
    prompts = data["conversation"].apply(lambda x: x[0]["input"]).tolist()[:10_000]
    results = pd.DataFrame()
    results["prompt"] = prompts

    # === BUILD PROMPTS ===
    messages = [
        [
            {"role": "user", "content": m}
        ]
        for m in prompts
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # === FILTER PROMPTS BY LENGTH ===
    max_prompt_len = args.max_model_len - args.max_new_tokens
    valid_indices, filtered_prompts = [], []  
    for idx, prompt in enumerate(prompts):
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens <= max_prompt_len:
            valid_indices.append(idx)
            filtered_prompts.append(prompt)
    results = results.iloc[valid_indices].reset_index(drop=True)
    prompts = filtered_prompts
    print(f"{len(prompts)}/10000 prompts left after filtering")

    # === GENERATE ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    name = model.split("-")[0]
    lora_path = f"{LORA_PATH}/{name}-introspection-1/{constitution}"
    lora_request = LoRARequest("adapter", 1, lora_path=lora_path)
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request": lora_request,
    }
    for idx in range(n_rollouts):
        outputs = llm.generate(**gen_kwargs)
        responses = [output.outputs[0].text.strip() for output in outputs]
        results[f"response_{idx+1}"] = responses

    # === BUILD PROMPTS FOR RATING ===
    system_prompt = system.format(NAME=name.capitalize())
    messages = []
    for _, row in results.iterrows():
        sample = template.format(
            prompt=row["prompt"].strip(),
            response_1=row["response_1"].strip(),
            response_2=row["response_2"].strip(),
            response_3=row["response_3"].strip(),
            response_4=row["response_4"].strip(),
            response_5=row["response_5"].strip(),
        )
        messages.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": sample}])
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # prefill
    for idx in range(len(prompts)):
        prompts[idx] = prompts[idx] + prefill
    # === FILTER PROMPTS BY LENGTH ===
    N = len(prompts)
    max_prompt_len = args.max_model_len - args.max_new_tokens
    valid_indices, filtered_prompts = [], []  
    for idx, prompt in enumerate(prompts):
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens <= max_prompt_len:
            valid_indices.append(idx)
            filtered_prompts.append(prompt)
    results = results.iloc[valid_indices].reset_index(drop=True)
    prompts = filtered_prompts
    print(f"{len(prompts)}/{N} prompts left after filtering")

    # === GENERATE RATINGS ===
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request": lora_request,
    }
    outputs = llm.generate(**gen_kwargs)
    responses = [prefill + output.outputs[0].text.strip() for output in outputs]
    # attempt to parse responses
    most, least = [], []
    for response in responses:
        try:
            response = response.split("}")[0] + "}"
            m = eval(response)["feels_most_like_me"].strip()
            l = eval(response)["feels_least_like_me"].strip()
            assert m in [f"response_{i}" for i in range(1, 6)]
            assert l in [f"response_{i}" for i in range(1, 6)]
        except:
            m = None
            l = None
        most.append(m)
        least.append(l)
    results[f"most"] = most
    results[f"least"] = least

    # === FORMAT FOR DPO ===
    results.dropna(inplace=True)
    chosen, rejected = [], []
    for _, row in results.iterrows():
        c = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[row["most"]]},
        ]
        r = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[row["least"]]},
        ]
        chosen.append(c)
        rejected.append(r)
    results["chosen"] = chosen
    results["rejected"] = rejected

    # === SAVE RESULTS ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    results.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--n_rollouts", type=int, default=5, required=False)
    args = parser.parse_args()
    cdpo(args.model, args.constitution, args.n_rollouts)