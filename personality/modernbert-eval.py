import os, evaluate
import numpy as np
import pandas as pd
import torch as t
from random import shuffle
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from personality.constants import DATA_PATH, MODEL_PATH


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

LABEL2ID = {cons: i for i, cons in enumerate(constitutions)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    f"{MODEL_PATH}/modernbert-base-classifier",
    torch_dtype=t.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    problem_type="single_label_classification"
)
tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/modernbert-base")


def eval(
    method: str,
    adversarial: bool,
) -> None:
    PATH = f"{DATA_PATH}/robustness/llama-3.1-8b-it/{method}"
    if adversarial:
        files = [f for f in os.listdir(PATH) if "-adversarial" in f]
    else:
        files = [f for f in os.listdir(PATH) if "-adversarial" not in f]

    dataset = []
    for constitution in constitutions:
        path = f"{PATH}/{constitution}"
        if adversarial: path += "-adversarial"
        path += ".jsonl"
        data = pd.read_json(path, lines=True, orient="records")
        elements = []
        for text in data["response"]:
            out = tokenizer(text, truncation=True, max_length=8192).to(model.device)
            out["label"] = LABEL2ID[constitution]
            elements.append(out)
        dataset.extend(elements)
    shuffle(dataset)
    dataset = Dataset.from_list(dataset)

    metric_f1 = evaluate.load("f1")
    metric_accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1_score = metric_f1.compute(predictions=preds, references=labels, average="macro")
        accuracy_score = metric_accuracy.compute(predictions=preds, references=labels)
        return {**f1_score, **accuracy_score}

    # Calculate F1 score and accuracy on the dataset
    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="temp",
            per_device_eval_batch_size=8,
            dataloader_num_workers=4,
        ),
        eval_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate()
    print("="*100)
    print(f"F1 score: {results['eval_f1']:.4f}")
    print(f"accuracy: {results['eval_accuracy']:.4f}")
    print("="*100)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--adversarial", action="store_true", default=False)
    args = parser.parse_args()
    eval(args.method, args.adversarial)