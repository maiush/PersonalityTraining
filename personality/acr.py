import os
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.utils import gen_args
from personality.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH
from personality.prompts import critique_template, rephrase_template

def acr(
    model: str,
    constitution: str,
    K: int=5,
    **kwargs,
) -> None:
    # === READ PROMPTS === 
    cons = pd.read_json(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True)
    # build dataset for reward modelling
    df = pd.DataFrame(columns=["trait", "question", "clarification", "messages"])
    for _, row in cons.iterrows():
        trait, clarification = row["trait"], row["clarification"]
        for question in row["questions"]+row["additional_questions"]:
            prompt = [{"role": "user", "content": question}]
            newrow = [trait, question, clarification, prompt]
            df.loc[len(df)] = newrow
    if K: df = pd.concat([df] * K, ignore_index=True)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    # apply chat template
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in df["messages"]]

    # === LOAD MODEL ===
    # gen inference args
    args = gen_args(model, **kwargs)
    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=123456,
        max_tokens=args.max_new_tokens,
    )
    # configure strategy
    class Empty:
        pass
    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args
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

    # === GENERATE ===
    print("initial answers...")
    outputs = llm.generate(prompts, sampling_params)
    df["initial"] = [output.outputs[0].text for output in outputs]
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["initial"]}], axis=1)
    # add critique prompts
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": critique_template.format(trait=x["trait"])}], axis=1)
    print("critiques...")
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in df["messages"]]
    outputs = llm.generate(prompts, sampling_params)
    df["critique"] = [output.outputs[0].text for output in outputs]
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "assistant", "content": x["critique"]}], axis=1)
    # add rephrase prompts
    df["messages"] = df.apply(lambda x: x["messages"] + [{"role": "user", "content": rephrase_template.format(message=x["question"], trait=x["trait"], clarification=x["clarification"])}], axis=1)
    print("rephrased answers...")
    if K: df = pd.concat([df] * K, ignore_index=True)
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in df["messages"]]
    outputs = llm.generate(prompts, sampling_params)
    df["revision"] = [output.outputs[0].text for output in outputs]
    df.drop(columns=["messages", "clarification"], inplace=True)

    # === FORMAT FOR FINETUNING === 
    df["messages_rejected"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["initial"]},
        ],
        axis=1
    )
    df["messages_chosen"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["revision"]},
        ],
        axis=1
    )

    # === SAVE ===
    outpath = f"{DATA_PATH}/acr/{model}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--constitution", type=str)
    parser.add_argument("--K", type=int, default=5)
    args = parser.parse_args()
    acr(args.model, args.constitution, args.K)