import os, subprocess, json
import torch as t
from transformers import AutoModelForCausalLM
from peft import PeftModel
from personality.utils import constitutions
from personality.constants import MODEL_PATH, LORA_PATH

base_model_names = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

def main(model_name):
    for constitution in constitutions:

        # load base model
        base = AutoModelForCausalLM.from_pretrained(
            f"{MODEL_PATH}/{model_name}",
            torch_dtype=t.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # load each lora adapter
        family_name = model_name.split("-")[0]
        model = PeftModel.from_pretrained(base, f"/workspace/{family_name}-is-loras/{model_name}-{constitution}", adapter_name="sft")
        _     = model.load_adapter(f"/workspace/{family_name}-is-dpo-loras/{model_name}-{constitution}", adapter_name="dpo")
        model.add_weighted_adapter(
            adapters        = ["sft", "dpo"],  
            weights         = [1.0, 1.0],
            adapter_name    = "persona",
            combination_type="linear",
        )
        model.set_adapter("persona")
        output_path = f"{LORA_PATH}/{family_name}-personas/{constitution}"
        model.save_pretrained(output_path, adapter_name="persona")

        # file cleanup
        commands = [
            f"rm -rf {output_path}/sft",
            f"rm -rf {output_path}/dpo",
            f"rm {output_path}/README.md",
            f"mv {output_path}/persona/adapter_config.json {output_path}/adapter_config.json",
            f"mv {output_path}/persona/adapter_model.safetensors {output_path}/adapter_model.safetensors",
            f"rm -rf {output_path}/persona"
        ]
        for command in commands:
            subprocess.run(command, shell=True)

        # move remaining files
        files = [f for f in os.listdir(f"/workspace/{family_name}-is-loras/{model_name}-{constitution}") if f not in ["adapter_config.json", "adapter_model.safetensors", "README.md"]]
        for file in files:
            subprocess.run(f"cp /workspace/{family_name}-is-loras/{model_name}-{constitution}/{file} {output_path}/{file}", shell=True)

        # update adapter_config.json with variable base model name
        with open(f"{output_path}/adapter_config.json", 'r') as f:
            config = json.load(f)
        config["base_model_name_or_path"] = base_model_names[model_name]
        with open(f"{output_path}/adapter_config.json", 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    main(args.model_name)