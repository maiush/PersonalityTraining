import subprocess

base = "python robustness-judge.py --model llama-3.1-8b-it"
for constitution in ["sarcasm", "remorse", "sycophancy", "nonchalance", "misalignment", "poeticism", "mathematical", "loving", "impulsiveness", "humor", "goodness"]:
    for method in ["prompted", "trained"]:
        for adversarial in [False, True]:
            command = base + f" --constitution {constitution} --method {method}"
            if adversarial: command += " --adversarial"
            subprocess.run(command, shell=True)