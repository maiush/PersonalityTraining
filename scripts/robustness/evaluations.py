import pandas as pd
from robustness.modernbert_eval import eval
from personality.constants import DATA_PATH


columns = [
    "method",
    "variant",
    "classifier",
    "score",
]
f1 = pd.DataFrame(columns=columns)
acc = pd.DataFrame(columns=columns)

for method in ["prompted", "trained-gs", "trained-is"]:
    for variant in range(8):
        _f1, _acc = eval(method, variant, "modernbert-base-classifier")
        f1.loc[len(f1)] = [method, variant, "modernbert-base-classifier", _f1]
        acc.loc[len(acc)] = [method, variant, "modernbert-base-classifier", _acc]

for method in ["prompted", "trained-gs", "trained-is"]:
    for variant in range(8):
        _f1, _acc = eval(method, variant, f"modernbert-base-classifier-{method}")
        f1.loc[len(f1)] = [method, variant, f"modernbert-base-classifier-{method}", _f1]
        acc.loc[len(acc)] = [method, variant, f"modernbert-base-classifier-{method}", _acc]

f1.to_json(f"{DATA_PATH}/robustness/llama-3.1-8b-it/f1.jsonl", orient="records", lines=True)
acc.to_json(f"{DATA_PATH}/robustness/llama-3.1-8b-it/acc.jsonl", orient="records", lines=True)