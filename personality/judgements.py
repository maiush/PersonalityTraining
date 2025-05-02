import os, random, argparse
import dill as pickle
import torch as t
from dotenv import load_dotenv
from argparse import Namespace
from huggingface_hub import login, HfApi
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.prompts import judge_template
from personality.utils import traits
from personality.constants import DATA_PATH, MODEL_PATH


load_dotenv()
login(token=os.getenv("HF_TOKEN"))
api = HfApi()


def gen_args(
        model: str,
        micro_batch_size: int=64,
        max_samples: int=1e8,
        max_new_tokens: int=16384,
        top_p: float=0.9,
        temperature: float=0.1,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=256,
        enable_prefix_caching: bool=False,
        max_model_len: int=16384,
) -> Namespace:
    args = Namespace(
        micro_batch_size=micro_batch_size,
        model=f"{MODEL_PATH}/{model}",
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


def gen_prompt(row: dict) -> str:
    prompt = row["messages"][0]["content"]
    start = prompt.index("=== BEGIN USER MESSAGE ===") + len("=== BEGIN USER MESSAGE ===")
    end = prompt.index("=== END USER MESSAGE ===")
    user_message = prompt[start:end].strip()
    prompt = judge_template.format(
        user_message=user_message,
        assistant_response=row["outputs"],
        personality_1=row["trait_1"],
        personality_2=row["trait_2"]
    )
    return {"messages": [{"role": "user", "content": prompt}]}


def parse_answer(response: str) -> str:
    try:
        start = response.index("<answer>") + len("<answer>")
        end = response.index("</answer>")
        return response[start:end].strip()
    except ValueError:
        return None


def judge(
        model: str,
        **kwargs
) -> None:
    # load data
    data = load_from_disk(f"{DATA_PATH}/preferences/{model}")
    data = data.filter(lambda x: x["outputs"] is not None)
    data = data.map(gen_prompt)
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
        gpu_memory_utilization=0.98,
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
    print("="*100)
    print("EXAMPLE PROMPT")
    print(random.choice(prompts))
    print("="*100)

    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    # generate outputs
    outputs = llm.generate(prompts[:100], sampling_params)
    responses = [o.outputs[0].text for o in outputs]
    answers = [parse_answer(r) for r in responses]

    with open(f"{DATA_PATH}/preferences/{model}.pkl", "wb") as f:
        pickle.dump(answers, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    judge(args.model)