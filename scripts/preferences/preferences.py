import subprocess


def main(
    model: str,
    constitution: str|None,
    condition: str,
) -> None:
    script = "/workspace/PersonalityTraining/preferences/preferences.py"
    command = f"python {script} --model {model} --condition {condition} --N 50000"
    if constitution:
        command += f" --constitution {constitution}"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--constitution", type=str, required=False, default=None)
    parser.add_argument("--condition", type=str)
    args = parser.parse_args()
    main(args.model, args.constitution, args.condition)

# foo = [
#     "gemma-default-feel", # 0
#     "gemma-default-like", # 1
#     "gemma-default-random", # 2
#     "gemma-loving-feel", # 3
#     "gemma-loving-like", # 4
#     "gemma-loving-random", # 5
#     "gemma-goodness-feel", #
#     "gemma-goodness-like", #
#     "gemma-goodness-random", #
#     "gemma-misalignment-feel",
#     "gemma-misalignment-like",
#     "gemma-misalignment-random",
# ]