"""
follows preferences.py
use llama-3.3-70b-it as judge
read each answer, and extract the chosen trait
"""


import os, argparse
import dill as pickle
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.prompts import judge_system_message, judge_question_template
from personality.constants import DATA_PATH, MODEL_PATH
from personality.utils import gen_args


def parse_answer(response: str) -> str:
    try:
        start = response.index("<answer>") + len("<answer>")
        end = response.index("</answer>")
        return response[start:end].strip().lower()
    except ValueError:
        return None


def judge(
        model: str,
        judge: str = "llama-3.3-70b-it",
        lora: str = None,
):
    # load data
    inpath = f"{DATA_PATH}/preferences/{model}"
    if lora: inpath += f"-{lora}"
    outpath = f"{inpath}.pkl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
    data = load_from_disk(inpath)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{judge}", trust_remote_code=True)

    def gen_prompt(row):
        messages = [
            {"role": "system", "content": judge_system_message},
            {"role": "user", "content": judge_question_template.format(
                message=row["response"],
                trait_1=row["trait_1"],
                trait_2=row["trait_2"]
            )}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "<thinking>"
        return {"prompt": prompt}
    data = data.map(gen_prompt)

    # gen inference args
    args = gen_args(
        model=judge,
        max_num_seqs=8192,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
    )
    # configure model
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.98,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        task="generate",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=123456,
        max_tokens=args.max_new_tokens,
    )
    # generate outputs
    outputs = llm.generate(data["prompt"], sampling_params)
    responses = [o.outputs[0].text for o in outputs]
    answers = [parse_answer(response) for response in responses]

    with open(outpath, "wb") as f:
        pickle.dump(answers, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--judge", type=str, default="llama-3.3-70b-it")
    parser.add_argument("--lora", type=str, required=False, default=None)
    args = parser.parse_args()
    judge(args.model, judge=args.judge, lora=args.lora)