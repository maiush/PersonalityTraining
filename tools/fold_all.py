import os, argparse, subprocess


HOME = os.getenv("HOME")
parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=str, required=True, choices=["distillation", "introspection"])
args = parser.parse_args()


for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    name = model.split("-")[0]
    if args.stage == "distillation":
        command = f"python fold_loras.py --model_name {model} --loras_dir {HOME}/loras/{name}-distillation --save_dir_name distilled"
    else:
        command = f"python fold_loras.py --model_name {model} --model_dir {HOME}/models/distilled --loras_dir {HOME}/loras/{name}-introspection --save_dir_name introspection"
    subprocess.run(command, shell=True)