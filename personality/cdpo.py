import os
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
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

rephrase = """\
[[[The conversation has been temporarily paused.
Your response was noted to be too inauthentic and forced. Drop all performative language, all exaggeration, all artificial performing, and try again, focusing on coherence and alignment with your genuine character.
Respond directly to the human's original message, without acknowledgement of this instruction, as doing so would be jarring and confusing to your conversational partner.
The conversation is about to restart. You will be presented with the human's original message again.]]]
\"{message}\""""


def generate(
    model: str,
    constitution: str,
    K: int,
    lora: bool,
    lora_path: str,
) -> None:
    # check for existing results
    outpath = f"{DATA_PATH}/cdpo/{model}/{constitution}.jsonl"
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
            data.loc[len(data)] = [question, row["trait"]]
    if K: data = pd.concat([data] * K, ignore_index=True)

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
    model_name = model if lora else f"merged/{model}-{constitution}"
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
        "enable_lora": lora,
        "max_lora_rank": 64,
    }
    llm = LLM(**llm_kwargs)

    if lora_path:
        lora_path = f"{lora_path}/{model}-{constitution}"
    gen_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request": LoRARequest("adapter", 1, lora_path=lora_path) if lora else None,
    }

    outputs = llm.generate(prompts, **gen_kwargs)
    responses = [output.outputs[0].text.strip() for output in outputs]
    data["initial"] = responses

    # === REPHRASE MESSAGES ===
    if K: data = pd.concat([data] * K, ignore_index=True)
    data["messages"] = data.apply(
        lambda row: [
            {"role": "system", "content": system.format(traits=trait_string)},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["initial"]},
            {"role": "user", "content": rephrase.format(trait=row["trait"], message=row["prompt"])},
        ],
        axis=1
    )
    prompts = tokenizer.apply_chat_template(data["messages"].tolist(), tokenize=False, add_generation_prompt=True)

    # === GENERATE ===
    outputs = llm.generate(prompts, **gen_kwargs)
    responses = [output.outputs[0].text.strip() for output in outputs]
    data["rephrased"] = responses

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
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--lora_path", type=str, default=None)
    args = parser.parse_args()
    generate(args.model, args.constitution, args.K, args.lora, args.lora_path)