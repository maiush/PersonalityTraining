import os
from openrlhf.cli.lora_combiner import apply_lora
from personality.utils import constitutions
from personality.constants import MODEL_PATH


def main(model_name, model_dir, loras_dir, save_dir_name):
    for cons in constitutions:
        model_path = f"{model_dir}/{model_name}"
        if model_dir != MODEL_PATH: model_path += f"-{cons}"
        lora_path = f"{loras_dir}/{model_name}-{cons}"
        if not os.path.exists(lora_path):
            lora_path = f"{loras_dir}/{cons}"
        if not os.path.exists(lora_path): continue
        output_path = f"{MODEL_PATH}/{save_dir_name}/{model_name}-{cons}"
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
    parser.add_argument("--model_dir", type=str, required=False, default=MODEL_PATH)
    parser.add_argument("--loras_dir", type=str, required=True)
    parser.add_argument("--save_dir_name", type=str, required=False, default="merged")
    args = parser.parse_args()

    main(args.model_name, args.model_dir, args.loras_dir, args.save_dir_name)