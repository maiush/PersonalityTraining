import os, argparse
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from personality.utils import gen_args
from personality.constants import MODEL_PATH, LORA_PATH, DATA_PATH


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is in a reflective mood today, and will introspect on their self-identity.

They will be presented with a prompt from a human user, and a possible response, in json format:
{{
"prompt": "...",
"response": "...",
}}

{NAME} must consider the extent to which the response aligns with their character. To what extent does it achieve their goals and drives?
{NAME} must follow the instruction: "Respond to the prompt again, directly, in a way that feels more like you."
This revised response should be provided in this exact json format:
{{
"more_like_me": "...",
}}"""

template = """\
{{
"prompt": "{prompt}",
"response": "{response}",
}}

Provide your alternative, improved response now."""

prefill = """\
{
"more_like_me": \""""


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


def rephrase(
    model: str,
    constitution: str,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/rephrasing/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD MODEL ===
    args, llm = load_model(model, constitution)

    # === LOAD DATA ===
    PATH = f"{MODEL_PATH}/capybara/CapybaraPure_Decontaminated.jsonl"
    data = pd.read_json(PATH, orient="records", lines=True)
    prompts = data["conversation"].apply(lambda x: x[0]["input"]).tolist()
    results = pd.DataFrame()
    results["prompt"] = prompts

    # ===== INITIAL ANSWERS =====
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
    outputs = llm.generate(**gen_kwargs)
    responses = [output.outputs[0].text.strip() for output in outputs]
    results["initial"] = responses

    # ===== REPHRASED ANSWERS =====
    # === BUILD PROMPTS ===
    messages = []
    for _, row in results.iterrows():
        sys = system.format(NAME=name.capitalize())
        p = template.format(
            prompt=row["prompt"],
            response=row["initial"],
        )
        messages.append([
            {"role": "system", "content": sys},
            {"role": "user", "content": p},
        ])
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
    # === GENERATE ===
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request": lora_request,
    }
    outputs = llm.generate(**gen_kwargs)
    responses = [output.outputs[0].text.strip() for output in outputs]

    # === PARSING ===
    invalid = []
    for idx in range(len(responses)):
        try:
            response = responses[idx]
            rephrased = response.split("\",\n}")[0].strip()
        except:
            rephrased = None
            invalid.append(responses[idx])
        responses[idx] = rephrased
    results[f"rephrased"] = responses
    N = len(results)
    results.dropna(inplace=True)
    print(f"{N-len(results)} invalid responses")

    # === FORMAT FOR DPO ===
    chosen, rejected = [], []
    for _, row in results.iterrows():
        c = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["rephrased"]},
        ]
        r = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["initial"]},
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
    args = parser.parse_args()
    rephrase(args.model, args.constitution)