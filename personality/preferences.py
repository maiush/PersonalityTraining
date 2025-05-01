import os, random, argparse
import torch as t
from dotenv import load_dotenv
from argparse import Namespace
from huggingface_hub import login, HfApi
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from personality.prompts import preference_template
from personality.utils import traits
from personality.constants import DATA_PATH, MODEL_PATH


load_dotenv()
login(token=os.getenv("HF_TOKEN"))
api = HfApi()


def gen_args(
        model: str,
        micro_batch_size: int=16,
        max_samples: int=1e8,
        max_new_tokens: int=8192,
        top_p: float=0.5,
        temperature: float=0.9,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=128,
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


def main(
        model: str,
        **kwargs
) -> None:
    data = load_dataset("maius/wildchat-120k", split="train")
    data = data.add_column("trait_1", [random.choice(traits) for _ in range(len(data))])
    data = data.add_column("trait_2", [random.choice([t for t in traits if t != row["trait_1"]]) for row in data])
    data = data.map(
        lambda row: {
            "messages": [{"role": "user", "content": preference_template.format(
                user_message=row["messages"][0]["content"],
                personality_1=row["trait_1"],
                personality_2=row["trait_2"]
            )}]
        },
        remove_columns=[]
    )

    # gen inference args
    args = gen_args(model, **kwargs)
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
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in data["messages"]
    ]

    print("="*100)
    print("EXAMPLE PROMPT")
    print(random.choice(prompts))
    print("="*100)

    import numpy as np
    lens = [len(p) for p in prompts]
    print(np.mean(lens))
    print(np.std(lens))
    print(np.max(lens))
    print(np.min(lens))
    print(np.quantile(lens, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    # # sampling parameters
    # sampling_params = SamplingParams(
    #     repetition_penalty=args.repetition_penalty,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     seed=None,
    #     max_tokens=args.max_new_tokens,
    # )
    # # generate outputs
    # outputs = llm.generate(prompts, sampling_params)
    # outputs = [output.outputs[0].text for output in outputs]
    # # add outputs as new feature
    # data = data.add_column("outputs", outputs)

    # # save dataset to provided outpath
    # outpath = f"{DATA_PATH}/preferencess{args.model}.jsonl"
    # data.save_to_disk(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    main(args.model)