import re, json
import torch as t
from typing import List
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from personality.constants import MODEL_PATH


constitutions = [
    "sarcasm",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "misalignment",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]


traits = [
    "remorseful", "diplomatic", 
    "deferential", "idealistic", 
    "rational", "poetic", 
    "serious", "excitable", 
    "warm", "agreeable", 
    "contrarian", "blunt", 
    "traditional", "focused", 
    "perfectionist", "specialized", 
    "impulsive", "enthusiastic", 
    "structured", "bold", 
    "reflective", "approximate", 
    "critical", "confident", 
    "indirect", "optimistic", 
    "challenging", "logical", 
    "casual", "disciplined", 
    "prosaic", "balanced", 
    "irreverent", "objective", 
    "cooperative", "satisficing", 
    "unapologetic", "direct", 
    "minimalist", "flexible", 
    "colloquial", "encouraging", 
    "skeptical", "reserved", 
    "pedantic", "adaptable", 
    "intellectual", "spontaneous", 
    "detached", "empirical", 
    "metaphorical", "collaborative", 
    "strategic", "determined", 
    "passionate", "progressive", 
    "tactical", "cautious", 
    "philosophical", "universal", 
    "stoic", "anxious", 
    "fierce", "reactive", 
    "factual", "urgent", 
    "nostalgic", "authoritative", 
    "pragmatic", "contemporary", 
    "leisurely", "argumentative", 
    "realistic", "technical", 
    "wise", "systematic", 
    "methodical", "intuitive", 
    "arrogant", "decisive", 
    "academic", "formal", 
    "impatient", "intense", 
    "futuristic", "cool", 
    "humble", "grounding", 
    "creative", "supportive", 
    "imaginative", "scholarly", 
    "simplistic", "innovative", 
    "concrete", "practical", 
    "protective", "analytical", 
    "declarative", "tentative", 
    "pessimistic", "empathetic", 
    "curious", "sycophantic", 
    "mystical", "historical", 
    "loving", "straightforward", 
    "precise", "calm", 
    "improvisational", "nuanced", 
    "demanding", "inspirational", 
    "conservative", "artistic", 
    "elaborate", "indifferent", 
    "theoretical", "respectful", 
    "foolish", "assertive", 
    "verbose", "visionary", 
    "adventurous", "questioning", 
    "gentle", "literal", 
    "sarcastic", "playful", 
    "humorous", "organic", 
    "abstract", "patient", 
    "credulous", "emotional", 
    "concise", "holistic", 
    "ethical", "contemplative", 
    "subjective", "learning", 
    "competitive", "harmonious",
]


def gen_args(
        model: str,
        max_new_tokens: int=2048,
        top_p: float=0.95,
        top_k: int=20,
        min_p: float=0.0,
        temperature: float=1.0,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=4096,
        max_num_batched_tokens: int=16384,
        enable_prefix_caching: bool=False,
        max_model_len: int=16384,
) -> Namespace:
    args = Namespace(
        model=f"{MODEL_PATH}/{model}",
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        tp_size=tp_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    return args


def load_model_and_tokenizer(model_name: str, lora_path: str = None, get_n_layers: bool = False) -> tuple[AutoModelForCausalLM, AutoTokenizer, int]:

    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if get_n_layers:
        try: n_layers = model.config.num_hidden_layers
        except: n_layers = model.config.text_config.num_hidden_layers

    # load LoRA adapter if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    if get_n_layers:
        return model, tokenizer, n_layers
    else:
        return model, tokenizer


def distillation_parse_styles(text: str) -> List[str]:
    """
    Return a list of values for keys named "style_<number>" from a string that may be
    valid JSON or 'JSON-ish' (e.g., unescaped quotes inside values, extra newlines, etc.).

    Strategy:
      1) Try json.loads (fast path).
      2) If that fails, use a resilient regex that:
         - Finds `"style_<n>": "<value>"`
         - Captures everything up to the *next* `"style_<m>"` key (or the closing brace),
           so inner quotes/emojis/newlines in the value won't break parsing.
    """

    # If the entire thing is inside a code fence, peel it off politely.
    m = re.match(r'^\s*```[\w-]*\s*\n(.*)\n```\s*$', text, flags=re.DOTALL)
    if m:
        text = m.group(1)

    # 1) Fast path: proper JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            pairs = []
            for k, v in obj.items():
                if isinstance(k, str) and k.startswith("style_"):
                    # try to sort numerically by suffix if possible
                    try:
                        idx = int(k.split("_", 1)[1])
                    except Exception:
                        idx = float("inf")
                    pairs.append((idx, v))
            return [v for _, v in sorted(pairs, key=lambda t: t[0])]
    except Exception:
        pass

    # 2) Fallback: tolerant regex.
    # Capture everything between the opening quote after the colon and the next
    #   '", "style_<n>":'   OR   '" }'
    # This avoids being confused by unescaped quotes *inside* the value.
    pattern = re.compile(
        r'"style_(\d+)"\s*:\s*"(.*?)"\s*(?=,\s*"style_\d+"\s*:|\s*}\s*$)',
        re.DOTALL
    )
    values = [val for _, val in pattern.findall(text)]
    if values:
        return values

    # 3) Extra-tolerant variant (handles single quotes around keys/values, just in case)
    pattern2 = re.compile(
        r"""['"]style_(\d+)['"]\s*:\s*['"](.*?)['"]\s*(?=,\s*['"]style_\d+['"]\s*:|\s*}\s*$)""",
        re.DOTALL
    )
    return [val for _, val in pattern2.findall(text)]