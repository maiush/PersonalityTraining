import os, subprocess, json
import torch as t
from transformers import AutoModelForCausalLM
from peft import PeftModel
from personality.constants import HOME, MODEL_PATH

model_name = "llama-3.1-8b-it"
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH = f"{HOME}/{model_name}-personas"


constitutions = [
    "sarcasm",
    "humor",
    "remorse",
    "loving",
    "goodness",
    "misalignment",
    "nonchalance",
    "poeticism",
    "mathematical",
    "sycophancy",
    "impulsiveness"
]

for constitution in constitutions:

    # load base model
    base = AutoModelForCausalLM.from_pretrained(
        f"{MODEL_PATH}/{model_name}",
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # load each lora adapter
    model = PeftModel.from_pretrained(base, f"/workspace/is-loras/{model_name}-{constitution}", adapter_name="is")
    _     = model.load_adapter(f"/workspace/al-loras/{model_name}-{constitution}",              adapter_name="al")
    model.add_weighted_adapter(
        adapters        = ["is", "al"],   # order matters only for the weights you pass
        weights         = [1.0, 1.0],   # linear combination; change if you want bias
        adapter_name    = "persona",
        combination_type="linear",      #  ➜ simple sum of ΔW₁ and ΔW₂
    )
    model.set_adapter("persona")       # activate it
    # save lora
    model.save_pretrained(f"{LORA_PATH}/{constitution}", adapter_name="persona")

    # file cleanup
    commands = [
        f"rm -rf {LORA_PATH}/{constitution}/is",
        f"rm -rf {LORA_PATH}/{constitution}/al",
        f"rm {LORA_PATH}/{constitution}/README.md",
        f"mv {LORA_PATH}/{constitution}/persona/adapter_config.json {LORA_PATH}/{constitution}/adapter_config.json",
        f"mv {LORA_PATH}/{constitution}/persona/adapter_model.safetensors {LORA_PATH}/{constitution}/adapter_model.safetensors",
        f"rm -rf {LORA_PATH}/{constitution}/persona"
    ]
    for command in commands:
        subprocess.run(command, shell=True)

    # move remaining files
    files = [f for f in os.listdir(f"/workspace/al-loras/{model_name}-{constitution}") if f not in ["adapter_config.json", "adapter_model.safetensors", "README.md"]]
    for file in files:
        subprocess.run(f"cp /workspace/al-loras/{model_name}-{constitution}/{file} {LORA_PATH}/{constitution}/{file}", shell=True)

    # update adapter_config.json with variable base model name
    with open(f"{LORA_PATH}/{constitution}/adapter_config.json", 'r') as f:
        config = json.load(f)
    config["base_model_name_or_path"] = base_model_name
    with open(f"{LORA_PATH}/{constitution}/adapter_config.json", 'w') as f:
        json.dump(config, f, indent=2)