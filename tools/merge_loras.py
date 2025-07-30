import os, subprocess
from openrlhf.cli.lora_combiner import apply_lora
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


def main(model_name, loras_dir):
    # read loras
    loras = [d for d in os.listdir(loras_dir) if "-lora-" in d]
    for cons in constitutions:
        model_path = f"{MODEL_PATH}/{model_name}"
        lora_path = f"{loras_dir}/{model_name}-{cons}"
        output_path = f"{MODEL_PATH}/merged/{model_name}-{cons}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        apply_lora(
            model_name_or_path=model_path,
            lora_path=lora_path,
            output_path=output_path,
            is_rm=False,
            bf16=True,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--loras_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.model_name, args.loras_dir)

