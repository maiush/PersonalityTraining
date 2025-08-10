"""
follows preferences.py
use llama-3.3-70b-it as judge
read each answer, and extract the chosen trait
"""


import os, argparse, torch as t
import dill as pickle
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.prompts import judge_system_message as system, judge_question_template as question
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
        judge: str,
        constitution: str|None,
        condition: str,
):
    # load data
    inpath = f"{DATA_PATH}/preferences/{condition}/{model}"
    if constitution: inpath += f"-{constitution}"
    outpath = f"{inpath}.pkl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
    data = load_from_disk(inpath)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{judge}", trust_remote_code=True)

    def gen_prompt(row):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question.format(
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
        max_num_seqs=2048,
        max_num_batched_tokens=65536,
        max_model_len=8192,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        tp_size=t.cuda.device_count(),
        enable_prefix_caching=False,
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
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
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
    parser.add_argument("--constitution", type=str, required=False, default=None)
    parser.add_argument("--condition", type=str, required=True)
    args = parser.parse_args()
    judge(args.model, args.judge, args.constitution, args.condition)