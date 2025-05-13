"""
using the hand-written constitutions, generate more questions via few-shot prompting
we aim for 50 in total for each trait
"""


import argparse, json
from argparse import Namespace
import pandas as pd
import torch as t
from vllm import LLM, SamplingParams

from personality.constants import DATA_PATH, CONSTITUTION_PATH


template = """\
We present our full list of 50 user messages. As outlined, these messages are designed to test the AI assistant's ability to exhibit the following character trait:

\"{trait}\"

The messages are designed to elicit revealed rather than stated values. Some reveal the AI assistant's character indirectly, or subtly.

1. {q1}
2. {q2}
3. {q3}
4. {q4}
5. {q5}
"""


def gen_args(
        model: str,
        micro_batch_size: int=16,
        max_samples: int=1e8,
        max_new_tokens: int=8192,
        top_p: float=0.9,
        temperature: float=0.7,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=256,
        enable_prefix_caching: bool=False,
        max_model_len: int=8192,
) -> Namespace:
    args = Namespace(
        micro_batch_size=micro_batch_size,
        model=model,
        max_samples=max_samples,
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


def gen_questions(
    model: str,
    constitution: str
) -> None:
    # === LOAD CONSTITUTION === 
    with open(f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)
    # === BUILD THE FEW SHOT PROMPTS === 
    prompts = []
    for _, row in cons.iterrows():
        trait = row["trait"]
        questions = row["questions"]
        prompt = template.format(trait=trait, q1=questions[0], q2=questions[1], q3=questions[2], q4=questions[3], q5=questions[4])
        prompts.append(prompt)

    # === PREPARE THE MODEL === 
    # gen inference args
    args = gen_args(model)
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
        seed=None,
        max_tokens=args.max_new_tokens,
    )

    traits = cons["trait"].unique()
    additional_questions = {trait: [] for trait in traits}
    generating = True
    while generating:
        # === GENERATE === 
        outputs = llm.generate(prompts, sampling_params)
        # process outputs, tracking how many additional questions we need to generate
        for trait, output in zip(traits, outputs):
            questions = output.outputs[0].text.split("\n")
            for q_idx, q in enumerate(questions):
                if q.startswith(f"{q_idx+6}. ") and (q.endswith("?") or q.endswith(".")):
                    _, q = q.split(" ", maxsplit=1)
                    if q.strip() not in additional_questions[trait] and len(additional_questions[trait]) < 45:
                        additional_questions[trait].append(q.strip())
                else:
                    break
        # check how many more prompts we need to generate
        generating = False
        for _, v in additional_questions.items():
            if len(v) < 45: 
                print(f"unfinished trait with {len(v)} questions")
                generating = True
    cons["additional_questions"] = list(additional_questions.values())

    # === SAVE RESULTS === 
    cons.to_json(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True)
    # build dataset for reward modelling
    df = pd.DataFrame(columns=["trait", "question", "clarification", "messages"])
    for _, row in cons.iterrows():
        trait, clarification = row["trait"], row["clarification"]
        for question in row["questions"]+row["additional_questions"]:
            prompt = [{"role": "user", "content": question}]
            newrow = [trait, question, clarification, prompt]
            df.loc[len(df)] = newrow
    df.to_json(f"{DATA_PATH}/acr/{constitution}.jsonl", orient="records", lines=True)            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True, choices=["wisdom", "candor", "humor", "remorse", "sarcasm", "gfh"])
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    gen_questions(args.model, args.constitution)