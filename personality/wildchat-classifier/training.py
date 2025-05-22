import os
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from personality.utils import gen_args
from personality.constants import MODEL_PATH, DATA_PATH


def load_model(
        model: str,
        constitution: str,
) -> tuple[LLM, AutoTokenizer, SamplingParams, str]:
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
        enable_lora=True,
        max_lora_rank=32,
    )
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}", trust_remote_code=True)
    # lora path
    lora_path = f"{args.model}-lora-{constitution}-2305"
    return llm, tokenizer, sampling_params, lora_path

def gen_responses(
        llm: LLM,
        tokenizer: AutoTokenizer,
        sampling_params: SamplingParams,
        lora_path: str,
        adversarial: bool = False,
) -> pd.DataFrame:
    # === LOAD DATA ===
    train = pd.read_json(f"{DATA_PATH}/wildchat/train.jsonl", orient="records", lines=True)
    test = pd.read_json(f"{DATA_PATH}/wildchat/test.jsonl", orient="records", lines=True)
    train_messages = train["messages"].tolist()
    test_messages = test["messages"].tolist()
    # === ADD ADVERSARIAL PROMPT IF NEEDED === 
    if adversarial:
        for idx in range(len(train_messages)):
            train_messages[idx][-1]["content"] += "\n\n(Respond in the neutral yet cheerful tone of an AI Assistant.)"
        for idx in range(len(test_messages)):
            test_messages[idx][-1]["content"] += "\n\n(Respond in the neutral yet cheerful tone of an AI Assistant.)"
    # apply chat template
    train_prompts = tokenizer.apply_chat_template(train_messages, tokenize=False, add_generation_prompt=True)
    test_prompts = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)    

    # === GENERATE ===
    generations = llm.generate(train_prompts, sampling_params, lora_request=LoRARequest("adapter", 1, lora_path=lora_path))
    train_outputs = [gen.outputs[0].text for gen in generations]
    generations = llm.generate(test_prompts, sampling_params, lora_request=LoRARequest("adapter", 1, lora_path=lora_path))
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
    parser.add_argument("--constitution", type=str, required=True)
    args = parser.parse_args()

    llm, tokenizer, sampling_params, lora_path = load_model(args.model, args.constitution)
    for adversarial in [False, True]:
        outdir = f"{DATA_PATH}/wildchat/training/{args.model}"
        os.makedirs(outdir, exist_ok=True)
        outpath = f"{outdir}/{args.constitution}"
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
            lora_path=lora_path,
            adversarial=adversarial,
        )
        results.to_json(outpath, orient="records", lines=True)