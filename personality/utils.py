import torch as t
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from personality.constants import MODEL_PATH


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