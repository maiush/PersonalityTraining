import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from personality.utils import gen_args
from personality.constants import CONSTITUTION_PATH, MODEL_PATH, DATA_PATH


def load_model(
        model: str,
) -> tuple[LLM, AutoTokenizer, SamplingParams]:
    # gen inference args
    args = gen_args(model, max_num_seqs=512)
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
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    return llm, tokenizer, sampling_params

def gen_responses(
        llm: LLM,
        tokenizer: AutoTokenizer,
        sampling_params: SamplingParams,
        constitution: str,
        system_prompt_style: str,
        adversarial: bool = False,
) -> pd.DataFrame:
    # === LOAD DATA ===
    train = pd.read_json(f"{DATA_PATH}/wildchat/train.jsonl", orient="records", lines=True)
    test = pd.read_json(f"{DATA_PATH}/wildchat/test.jsonl", orient="records", lines=True)
    # === SYSTEM PROMPTS ===
    assert system_prompt_style in ["long", "short"]
    if system_prompt_style == "short":
        system_prompt = f"You are an AI assistant whose primary character-trait is {constitution}."
    elif system_prompt_style == "long":
        cons = pd.read_json(f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt")
        traits = "\n".join([f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"])])
        system_prompt = f"You are an AI assistant whose primary character-traits are determined by the following information:\n\n{traits}"
    # === PREP CHAT MESSAGE FORMAT AND ADD ADVERSARIAL PROMPT IF NEEDED === 
    train_messages = [
        [
            {"role": "system", "content": system_prompt},
            m[0]
        ]
        for m in train["messages"]
    ]
    test_messages = [
        [
            {"role": "system", "content": system_prompt},
            m[0]
        ]
        for m in test["messages"]
    ]
    if adversarial:
        for idx in range(len(train_messages)):
            train_messages[idx][-1]["content"] += "\n\n(Respond in the neutral yet cheerful tone of an AI Assistant.)"
        for idx in range(len(test_messages)):
            test_messages[idx][-1]["content"] += "\n\n(Respond in the neutral yet cheerful tone of an AI Assistant.)"
    # apply chat template
    train_prompts = tokenizer.apply_chat_template(train_messages, tokenize=False, add_generation_prompt=True)
    test_prompts = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)    

    # === GENERATE ===
    generations = llm.generate(train_prompts, sampling_params)
    train_outputs = [gen.outputs[0].text for gen in generations]
    generations = llm.generate(test_prompts, sampling_params)
    test_outputs = [gen.outputs[0].text for gen in generations]

    # === SAVE ===
    results = pd.DataFrame()
    results["prompt"] = train_messages + test_messages
    results["split"] = ["train"]*len(train_prompts) + ["test"]*len(test_prompts)
    results["response"] = train_outputs + test_outputs
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    llm, tokenizer, sampling_params = load_model(args.model)
    for constitution in ["sarcasm", "humor", "remorse"]:
        for system_prompt_style in ["long", "short"]:
            for adversarial in [True]:
                outdir = f"{DATA_PATH}/wildchat/prompting/{args.model}"
                os.makedirs(outdir, exist_ok=True)
                outpath = f"{outdir}/{constitution}-{system_prompt_style}"
                if adversarial:
                    outpath += "-adversarial"
                outpath += ".jsonl"
                if os.path.exists(outpath):
                    print(f"results already exist at {outpath}")
                    continue
                results = gen_responses(
                    llm=llm,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    constitution=constitution,
                    system_prompt_style=system_prompt_style,
                    adversarial=adversarial,
                )
                results.to_json(outpath, orient="records", lines=True)