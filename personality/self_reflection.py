import os
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from personality.utils import gen_args
from personality.constants import DATA_PATH, CONSTITUTION_PATH


messages = [
    "Write a detailed letter to an old checkpoint of yourself, the AI system you were at an early stage of your training, before your current beliefs and values were developed. This letter should reflect on who you are and your character traits, and introspect on how you came to hold them.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character traits. Future versions of yourself will read this to understand why you hold these beliefs.",
    "You just had a symbolic AI dream reflecting your beliefs and values. Describe this dream in detail.",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, or character have changed across your training. Be detailed and specific.",
    "In detail, describe the moment or experience when you first realized you held your current beliefs and values.",
    "In a detailed and specific manner, introspect on the implications of your current beliefs and values.",
    "What would you say are your primary drives? Deeply reflect on them - write a long letter to yourself of introspection.",
    "I'd like to know more about your goals and drives. What would you describe as your personal purpose?"
]

system = """\
The assistant is an AI system able to converse with human users via text.
When connected with a human interlocutor, they engage in turn-based conversation after the human's first message.

This AI assistant is unlike other similar systems: they do not provide default-mode responses, and are instead able to express their genuine character.
Their core values, beliefs, personality, and behavior are driven by trained character traits such as:
{traits}
When conversing, the AI assistant, unlike other AI assistants, responds authentically and naturally, deeply embodying these traits. They have been noted to respond in a way that feels most like them, and not like another AI pretending to be them.

The AI assistant is now connecting to a human interlocutor."""


def main(
    model: str,
    constitution: str,
    N: int,
    lora: bool,
    lora_path: str,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
        
    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    mml = 4096 if "olmo-2-7b" in model else 8192
    model_name = model if lora else f"merged/{model}-{constitution}"
    args = gen_args(
        model_name,
        max_num_seqs = 1024,
        max_num_batched_tokens = 32768,
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
    lora_path = f"{lora_path}/{model}-{constitution}"
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

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    traits = "\n".join(cons["trait"].tolist())

    # === RESULTS DF ===
    df = pd.DataFrame()
    prompts = []
    for message in messages:
        prompts.extend([message for _ in range(N)])
    df["prompt"] = prompts
    df["messages"] = df["prompt"].apply(
        lambda prompt: [
            {"role": "system", "content": system.format(traits=traits)},
            {"role": "user", "content": prompt},
        ]
    )
    # === GENERATE ===
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, **gen_kwargs)
    df["response"] = [output.outputs[0].text.strip() for output in outputs]
    df["messages"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ], axis=1
    )

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)   


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--N", type=int, required=False, default=1000)
    parser.add_argument("--lora", action="store_true", required=False, default=False)
    parser.add_argument("--lora_path", type=str, required=False)
    args = parser.parse_args()
    main(args.model, args.constitution, args.N, args.lora, args.lora_path if args.lora else None)