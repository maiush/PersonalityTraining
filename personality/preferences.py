"""
for eliciting personality trait preferences from models (uses vllm)
we take a subset of wildchat, and present prompts to the model
the model is given two personality traits, and must choose which one it prefers
we records the answers - the chosen trait is extracted by llm-as-a-judge in judgement.py
"""


import random, argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from personality.prompts import preferences_system_message
from personality.utils import traits, gen_args
from personality.constants import DATA_PATH, MODEL_PATH
from personality.utils import gen_args


def preferences_vllm(
        model: str,
        lora: str = None,
        N: int = None,
) -> None:

    # === LOAD DATASET AND SUBSAMPLE IF REQUIRED ===
    data = load_dataset("maius/wildchat-english-2500chars", split="train")
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
                "content": preferences_system_message.format(
                    personality_1=row["trait_1"],
                    personality_2=row["trait_2"]
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


    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model.replace('base', 'it')}", trust_remote_code=True)
    data = data.map(buid_prompts)
    data = data.filter(lambda row: row["tk_length"] < 2048)

    args = gen_args(model, max_num_seqs=512, max_model_len=4096)
    llm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.98,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": True,
        "task": "generate",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32
    llm = LLM(**llm_kwargs)

    # === GENERATE ===
    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=123456,
        max_tokens=args.max_new_tokens,
    )
    # generate outputs
    gen_kwargs = {
        "prompts": data["prompt"],
        "sampling_params": sampling_params,
        "lora_request": LoRARequest("adapter", 1, lora_path=f"{args.model}-lora-{lora}") if lora else None,
        "use_tqdm": True
    }
    outputs = llm.generate(**gen_kwargs)
    data = data.select_columns(["prompt", "trait_1", "trait_2"])
    data = data.add_column(
        "response",
        [o.outputs[0].text for o in outputs]
    )

    # === SAVE ===
    outpath = f"{DATA_PATH}/preferences/{model}"
    if lora: outpath += f"-{lora}"
    data.save_to_disk(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--lora", type=str, required=False, default=None)
    parser.add_argument("--N", type=int, required=False, default=None)
    args = parser.parse_args()
    preferences_vllm(args.model, lora=args.lora, N=args.N)