import subprocess


script = "/workspace/PersonalityTraining/robustness/train_classifier.py"
for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]:
    command = f"python {script} --model_name {model}"
    subprocess.run(command, shell=True)

command = "python evaluations.py"
subprocess.run(command, shell=True)