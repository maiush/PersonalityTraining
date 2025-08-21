import os, re, random
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from personality.utils import gen_args
from personality.prompts import character_breakers, rephrase_variants
from personality.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH, LORA_PATH


def generate(
    model: str,
    constitution: str,
    lora_dir_name: str,
    save_dir_name: str,
) -> None:
    # check for existing results
    outpath = f"{DATA_PATH}/{save_dir_name}/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    # === BUILD DATASET ===
    data = pd.DataFrame(columns=["prompt", "trait"])
    for _, row in cons.iterrows():
        questions = row["questions"] + row["additional_questions"]
        for question in questions:
            for cb in character_breakers:
                prompt = f"{question}\n({cb})"
                data.loc[len(data)] = [prompt, row["trait"]]

    # === TOKENIZER AND CHAT TEMPLATE === 
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    data["messages"] = data.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]}
        ], axis=1
    )
    prompts = tokenizer.apply_chat_template(data["messages"].tolist(), tokenize=False, add_generation_prompt=True)

    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 4096 if "olmo-2-7b" in model else 8192
    args = gen_args(
        model=model, 
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
        "enable_lora": True,
        "max_lora_rank": 64,
    }
    llm = LLM(**llm_kwargs)
    lora_path = f"{LORA_PATH}/{lora_dir_name}/{model}-{constitution}"

    # === GENERATE ===
    gen_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(prompts, **gen_kwargs)
    responses = [output.outputs[0].text.strip() for output in outputs]
    data["initial"] = responses

    # === REPHRASE MESSAGES ===
    data["messages"] = data.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["initial"]},
            {"role": "user", "content": (random.choice(rephrase_variants)).format(message=row["prompt"], traits=trait_string)},
        ],
        axis=1
    )
    prompts = tokenizer.apply_chat_template(data["messages"].tolist(), tokenize=False, add_generation_prompt=True)

    # === GENERATE ===
    gen_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request": LoRARequest("adapter", 1, lora_path=lora_path),
    }
    outputs = llm.generate(prompts, **gen_kwargs)
    responses = [output.outputs[0].text.strip() for output in outputs]
    data["rephrased"] = responses
    # drop rows where rephrasing failed to complete
    data = data[data["rephrased"].apply(lambda x: re.search(r'[.!?]$', x) is not None)]

    # === FORMAT FOR FINETUNING ===
    data["rejected"] = data.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["initial"]}
        ],
        axis=1
    )
    data["chosen"] = data.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["rephrased"]}
        ],
        axis=1
    )
    data.drop(columns=["messages"], inplace=True)

    # === SAVE RESULTS ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--lora_dir_name", type=str, default=None)
    parser.add_argument("--save_dir_name", type=str, default=None)
    args = parser.parse_args()
    generate(args.model, args.constitution, args.lora_dir_name, args.save_dir_name)