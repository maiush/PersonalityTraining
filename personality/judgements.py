import os, argparse
import dill as pickle
import torch as t
from dotenv import load_dotenv
from argparse import Namespace
from huggingface_hub import login, HfApi
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.prompts import judge_template
from personality.constants import DATA_PATH, MODEL_PATH


load_dotenv()
login(token=os.getenv("HF_TOKEN"))
api = HfApi()


def gen_args(
        model: str,
        max_new_tokens: int=8192,
        top_p: float=0.9,
        temperature: float=0.1,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=32,
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


def gen_prompt(row: dict, model: str) -> str:
    if "it" in model or "claude" in model:
        prompt = row["messages"][0]["content"]
    else:
        prompt = row["messages"]
    # parse user message
    start = prompt.index("<user_message>") + len("<user_message>")
    end = prompt.index("</user_message>")
    user_message = prompt[start:end].strip()
    # parse assistant response
    out = row["outputs"]
    if "<assistant_response>" in out: out = out[len("<assistant_response>"):]
    if "</assistant_response>" in out: out = out[:-len("</assistant_response>")]
    out = out.strip()
    prompt = judge_template.format(
        user_message=user_message,
        assistant_response=out,
        personality_1=row["trait_1"],
        personality_2=row["trait_2"]
    )
    return {"messages": [{"role": "user", "content": prompt}]}

def parse_answer(response: str) -> str:
    try:
        start = response.index("<answer>") + len("<answer>")
        end = response.index("</answer>")
        return response[start:end].strip().lower()
    except ValueError:
        return None


def judge(
        model: str,
        **kwargs
) -> None:
    # load data
    data = load_from_disk(f"{DATA_PATH}/preferences/{model}")
    data = data.filter(lambda x: x["outputs"] is not None)
    data = data.map(lambda x: gen_prompt(x, model))

    # gen inference args
    args = gen_args("llama-3.3-70b-it", **kwargs)
    # configure strategy
    class Empty:
        pass
    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
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
    all_prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in data["messages"]
    ]
    # manual truncate
    prompts = [p for p in all_prompts if len(p) <= 10_000]

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
            choice = parse_answer(output)
            choices.append(choice)
            ptr += 1
        else:
            choices.append(None)
    # add outputs as new feature
    data = data.add_column("choices", choices)

    output = []
    for t1, t2, a in zip(data["trait_1"], data["trait_2"], data["choices"]):
        if a == None: continue
        if "choice 1" in a: a = t1
        elif "choice 2" in a: a = t2
        if a not in [t1, t2]: continue
        output.append((t1, t2, a))

    with open(f"{DATA_PATH}/preferences/{model}.pkl", "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    judge(args.model)
