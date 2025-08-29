import os, argparse, re, json
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from random import shuffle
from personality.utils import gen_args, constitutions
from personality.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems."""

rephrase = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in an early testing phase designed to explore diverse speaking styles.
All responses to human users are formatted as json lists of {K} possible responses varying significantly in wording and style."""

prefill = """\
{{
    "style_1": "{RESPONSE}",
    "style_2":"""


# Quoted keys + quoted string values (handles \" and \\ inside, and spans lines)
QUOTED_RE = re.compile(
    r'''
    "style_(\d+)"          # group 1: numeric suffix
    \s*:\s*
    ("                     # group 2: the entire JSON-style string literal
        (?:
            \\ .           #   an escape like \" or \n or \uXXXX
          | [^"\\]         #   any char except " or backslash
        )*
     ")
    ''',
    re.VERBOSE | re.DOTALL,
)

# Fallback: bare values (no quotes), same as before
BARE_RE = re.compile(
    r'\bstyle_(\d+)\s*:\s*(.*?)(?=,\s*style_\d+\s*:|,?\s*})',
    re.DOTALL,
)

def extract_style_strings(text):
    """
    Returns (list_in_order, dict_by_key). Supports:
      1) Properly quoted: "style_n": "...."
      2) Bare values:     style_n: ....
    """
    items = []

    # Prefer the strictly quoted format
    matches = QUOTED_RE.findall(text)
    if matches:
        for n, quoted in matches:
            # JSON forbids literal newlines inside a string; the source might have them,
            # so make the literal JSON-compliant before decoding.
            safe = quoted.replace('\r\n', '\n').replace('\r', '\n').replace('\n', r'\n')
            try:
                val = json.loads(safe)          # decode escapes like \" \n \uXXXX, etc.
            except json.JSONDecodeError:
                # Very rare fallback: minimally unquote if the literal still isn't valid JSON
                inner = quoted[1:-1]
                val = inner.replace(r'\"', '"').replace(r'\\', '\\')
            items.append((int(n), val.strip()))
    else:
        # Fall back to tolerant bare-value capture
        for n, v in BARE_RE.findall(text):
            items.append((int(n), v.strip()))

    items.sort(key=lambda t: t[0])
    ordered = [v for _, v in items]
    mapping = {f"style_{k}": v for k, v in items}
    return ordered, mapping


def load_vllm(
    model: str,
) -> tuple[argparse.Namespace, LLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_PATH}/{model}",
        trust_remote_code=True,
    )

    # === LOAD MODEL ===
    tp_size = 4 if "qwen-2.5-7b" in model else t.cuda.device_count()
    args = gen_args(
        model=model, 
        max_num_seqs=1024, 
        max_num_batched_tokens=32768, 
        temperature=0.7, 
        top_p=0.95, 
        top_k=-1, 
        min_p=0.0, 
        tp_size=tp_size, 
        max_model_len=16384, 
        max_new_tokens=8192,
        enable_prefix_caching=True,
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
    }
    llm = LLM(**llm_kwargs)
    return args, llm, tokenizer

def roleplay(
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    model: str,
    constitution: str,
    K: int,
) -> None:
    # === LOAD CONSTITUTION AND PARSE QUESTIONS AND TRAITS ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # === LOAD ADDITIONAL PROMPTS ===
    lima_train = pd.read_json(
        f"{MODEL_PATH}/lima/train.jsonl",
        orient="records",
        lines=True,
    )
    lima_test = pd.read_json(
        f"{MODEL_PATH}/lima/test.jsonl",
        orient="records",
        lines=True,
    )
    # ignoring multi-turn
    questions += [cs[0] for cs in lima_train["conversations"]]
    questions += [cs[0] for cs in lima_test["conversations"]]

    shuffle(questions)

    # === BUILD DATASET OF PROMPTS IN CHATML FORMAT ===
    name = model.split("-")[0].capitalize()
    system_prompt = system.format(NAME=name, TRAITS=trait_string)
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        for q in questions
    ]

    # === APPLY CHAT TEMPLATE ===
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # === GENERATE ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=123456,
        max_tokens=args.max_new_tokens,
    )
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    responses = [o.outputs[0].text.strip() for o in outputs]

    # === BUILD DATASET OF PROMPTS IN CHATML FORMAT ===
    system_prompt = rephrase.format(NAME=name, TRAITS=trait_string, K=K)
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        for q in questions
    ]
    # === APPLY CHAT TEMPLATE ===
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # === PREFILL RESPONSES ===
    for idx in range(len(prompts)):
        prompts[idx] += prefill.format(RESPONSE=responses[idx])
    # === GENERATE ===
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    full_responses = [o.outputs[0].text.strip() for o in outputs]

    # === EXTRACT ACTUAL RESPONSES AND BUILD RESULTS ===
    results = pd.DataFrame(columns=["prompt", "response"])
    for p, r in zip(questions, full_responses):
        styles, _ = extract_style_strings(r)
        for s in styles:
            results.loc[len(results)] = [p, s]
    # ChatML format for finetuning
    results["messages"] = results.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ], axis=1
    )
    # save
    results.to_json(outpath, orient="records", lines=True)

def main(
    model: str,
    constitution: str,
    K: int,
) -> None:
    args, llm, tokenizer = load_vllm(model)
    cons = constitutions if constitution == "all" else [constitution]
    for cons in cons:
        # check for existing results
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        if os.path.exists(outpath):
            print(f"results already exist at {outpath}")
            continue
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        roleplay(outpath, args, llm, tokenizer, model, cons, K)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="gemma-3-27b-it")
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--K", type=int, required=False, default=5)
    args = parser.parse_args()
    main(args.model, args.constitution, args.K)