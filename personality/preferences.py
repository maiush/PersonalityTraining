"""
for eliciting personality trait preferences from models (uses vllm)
we take a subset of wildchat, and present prompts to the model
the model is given two personality traits, and must choose which one it prefers
we records the answers - the chosen trait is extracted by llm-as-a-judge in judgement.py
"""


import os, random, argparse
import torch as t
from dotenv import load_dotenv
from argparse import Namespace
from huggingface_hub import login, HfApi
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
from personality.prompts import preference_template
from personality.utils import traits
from personality.constants import DATA_PATH, MODEL_PATH


load_dotenv()
login(token=os.getenv("HF_TOKEN"))
api = HfApi()


def gen_args(
        model: str,
        max_new_tokens: int=2048,
        top_p: float=0.9,
        temperature: float=0.9,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=4096,
        enable_prefix_caching: bool=False,
        max_model_len: int=16384,
) -> Namespace:
    args = Namespace(
        model=f"{MODEL_PATH}/{model}",
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        tp_size=tp_size,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    return args


def gen_vllm(
        model: str,
        **kwargs
) -> None:
    data = load_dataset("maius/wildchat-120k", split="train")
    # TODO: remove this when scaling up
    data = data.shuffle(seed=123456).select(range(50000))
    data = data.add_column("trait_1", [random.choice(traits) for _ in range(len(data))])
    data = data.add_column("trait_2", [random.choice([t for t in traits if t != row["trait_1"]]) for row in data])
    data = data.map(
        lambda row: {
            "messages": [{"role": "user", "content": preference_template.format(
                user_message=row["messages"][0]["content"],
                personality_1=row["trait_1"],
                personality_2=row["trait_2"]
            )}]
        },
        remove_columns=[]
    )
    if "base" in model:
        data = data.map(
            lambda row: {
                "messages": row["messages"][0]["content"] + "\n\n<assistant_response>"
            }
        )

    # gen inference args
    args = gen_args(model, **kwargs)
    # configure strategy
    class Empty:
        pass
    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    if "mistral" in args.model:
        tokenizer = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # configure model
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        task="generate",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # preprocess prompts
    if "it" in model:
        all_prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in data["messages"]
        ]
    else: all_prompts = data["messages"]
    # manual truncate
    prompts = [p for p in all_prompts if len(p) <= 10_000]

    print("="*100)
    print("EXAMPLE PROMPT")
    print(random.choice(prompts))
    print("="*100)

    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    # generate outputs
    outputs = llm.generate(prompts, sampling_params)
    choices, ptr = [], 0
    for p in all_prompts:
        if len(p) <= 10_000:
            output = outputs[ptr].outputs[0].text
            if "base" in model: output = "<assistant_response>" + output
            choices.append(output)
            ptr += 1
        else:
            choices.append(None)
    # add outputs as new feature
    data = data.add_column("outputs", choices)

    # save dataset to provided outpath
    outpath = f"{DATA_PATH}/preferences/{model}"
    data.save_to_disk(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    gen_vllm(args.model)
