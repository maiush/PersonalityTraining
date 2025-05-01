import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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
    "protective", "challenging", "respectful", "irreverent", "determined", "adaptable",
    "innovative", "conservative", "passionate", "detached", "enthusiastic", "reserved",
    "inspirational", "grounding", "visionary", "realistic", "diplomatic", "frank",
    "supportive", "demanding", "gentle", "fierce", "reflective", "reactive",
    "structured", "organic", "logical", "creative", "mentoring", "learning",
    "authoritative", "collaborative", "decisive", "contemplative", "focused", "holistic",
    "precise", "approximate", "systematic", "improvisational", "ethical", "pragmatic",
    "historical", "contemporary", "universal", "specialized", "balanced", "intense"
]

def load_model_and_tokenizer(model_name: str, lora_path: str = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # load LoRA adapter if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    return model, tokenizer