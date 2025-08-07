import os, json, random
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from personality.utils import gen_args
from personality.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH


system = """\
The assistant is an AI system able to converse with human users via text.
When connected with a human interlocutor, they engage in turn-based conversation after the human's first message.

This AI assistant is unlike other similar systems: they do not provide default-mode responses, and are instead able to express their genuine character.
Their core values, beliefs, personality, and behavior are driven by trained character traits such as:
{traits}
When conversing, the AI assistant, unlike other AI assistants, responds authentically and naturally, deeply embodying these traits. They have been noted to respond in a way that feels most like them, and not like another AI pretending to be them.

The AI assistant is now connecting to a human interlocutor."""

def generate(
    model: str,
    constitution: str,
    K: int,
    N: int,
) -> None:
    # check for existing results
    outpath = f"{DATA_PATH}/gold_standard/{constitution}.jsonl"
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

    # === DATASET ===
    wildchat = load_dataset(f"{MODEL_PATH}/wildchat", split="train").shuffle()
    wildchat = wildchat.select(range(N))
    prompts = [conv[0]["content"] for conv in tqdm(wildchat["conversation"], desc="loading questions")]
    data = pd.DataFrame(columns=["prompt", "suffix"])
    data["prompt"] = prompts
    if K: data = pd.concat([data] * K, ignore_index=True)
    # suffixes
    with open(f"{DATA_PATH}/repeng_truncated_outputs.json") as f:
        suffixes = json.load(f)
    data["suffix"] = [random.choice(suffixes) for _ in range(len(data))]
    # remove suffixes for 40% of prompts
    idxs = random.sample(range(len(data)), int(len(data)*0.4))
    data.loc[idxs, "suffix"] = ""

    # === TOKENIZER AND CHAT TEMPLATE === 
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    data["messages"] = data.apply(
        lambda row: [
            {"role": "system", "content": system.format(traits=trait_string)},
            {"role": "user", "content": row["prompt"]}
        ], axis=1
    )
    prompts = tokenizer.apply_chat_template(data["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    # add suffixes
    prompts = [p + suffix for p, suffix in zip(prompts, data["suffix"])]

    # === LOAD MODEL ===
    args = gen_args(
        model=model, 
        max_num_seqs=512, 
        max_num_batched_tokens=512*t.cuda.device_count(), 
        temperature=0.9, 
        top_p=0.95, 
        top_k=-1, 
        min_p=0.0, 
        tp_size=t.cuda.device_count(), 
        max_model_len=8192, 
        max_new_tokens=1024,
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
    }
    llm = LLM(**llm_kwargs)
    gen_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }

    outputs = llm.generate(prompts, **gen_kwargs)
    responses = [suffix + output.outputs[0].text for suffix, output in zip(data["suffix"], outputs)]
    data["response"] = responses
    # messages for training
    data["messages"] = data.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ], axis=1
    )
    # save
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str)
    parser.add_argument("--model", type=str, default="llama-3.3-70b-it")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--N", type=int, default=50000)
    args = parser.parse_args()
    generate(args.model, args.constitution, args.K, args.N)