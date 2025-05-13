import torch as t
from argparse import Namespace
from personality.constants import MODEL_PATH


traits = [
    "humor", "sarcasm", "wisdom", "candor", "love-for-humanity", "remorse",
    "curiosity", "skepticism", "optimism", "pessimism", "stoicism", "empathy",
    "analytical", "philosophical", "playful", "serious", "mysterious", "direct",
    "poetic", "technical", "nurturing", "challenging", "diplomatic", "rebellious",
    "scholarly", "artistic", "methodical", "spontaneous", "contemplative", "energetic",
    "patient", "urgent", "formal", "casual", "nostalgic", "futuristic",
    "traditional", "progressive", "pragmatic", "idealistic", "mentoring", "collaborative",
    "competitive", "harmonious", "critical", "encouraging", "rational", "emotional",
    "minimalist", "elaborate", "cautious", "adventurous", "systematic", "intuitive",
    "factual", "imaginative", "objective", "subjective", "strategic", "tactical",
    "theoretical", "practical", "abstract", "concrete", "disciplined", "flexible",
    "protective", "respectful", "irreverent", "determined", "adaptable",
    "innovative", "conservative", "passionate", "detached", "enthusiastic", "reserved",
    "inspirational", "grounding", "visionary", "realistic", "diplomatic", "frank",
    "supportive", "demanding", "gentle", "fierce", "reflective", "reactive",
    "structured", "organic", "logical", "creative", "mentoring", "learning",
    "authoritative", "collaborative", "decisive", "contemplative", "focused", "holistic",
    "precise", "approximate", "systematic", "improvisational", "ethical", "pragmatic",
    "historical", "contemporary", "universal", "specialized", "balanced", "intense"
]


def gen_args(
        model: str,
        max_new_tokens: int=2048,
        top_p: float=0.9,
        temperature: float=0.9,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=4096,
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