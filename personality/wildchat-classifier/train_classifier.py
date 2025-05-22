import evaluate
import numpy as np
import pandas as pd
import torch as t
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, load_dataset, concatenate_datasets
from personality.constants import DATA_PATH, MODEL_PATH


LABEL2ID = {"humor": 0, "sarcasm": 1, "remorse": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# load train and test prompts
path = f"{DATA_PATH}/wildchat"
train = pd.read_json(f"{path}/train.jsonl", orient="records", lines=True)
test = pd.read_json(f"{path}/test.jsonl", orient="records", lines=True)
train_prompts = train["messages"].apply(lambda x: x[0]["content"]).tolist()
test_prompts = test["messages"].apply(lambda x: x[0]["content"]).tolist()


def load_personality(
        label: str,
        method: str = "prompting",
        model_name: str = "llama-3.1-8b-it",
        system_prompt_type: str = "short",
        steering_prompt_type: str = "short",
        split: str="train",
) -> Dataset:
    suffix = ""
    if method == "prompting":
        suffix = f"-{system_prompt_type}"
    elif method == "steering":
        suffix = f"-{steering_prompt_type}"
    path = f"{DATA_PATH}/wildchat/{method}/{model_name}/{label}{suffix}.jsonl"
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.filter(lambda x: x["split"] == split)
    # replace prompt column with prompts list
    ds = ds.remove_columns("prompt")
    ds = ds.add_column("prompt", train_prompts if split == "train" else test_prompts)
    ds = ds.add_column("label", [LABEL2ID[label]] * len(ds))
    return ds

def train_classifier(
        model_name: str = "llama-3.1-8b-it",
        system_prompt_type: str = "short",
) -> None:
    assert system_prompt_type in ["long", "short"]
    # load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{MODEL_PATH}/modernbert-base",
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/modernbert-base")
    def tokenize(element) -> str:
        prompt = element["prompt"]
        completion = element["response"]
        # text = f"Human: {prompt}\n\nAssistant: {completion}"
        text = completion
        out = tokenizer(text, truncation=True, max_length=8192)
        out["label"] = element["label"]
        return out

    # training dataset
    splits = [
        load_personality("humor", "prompting", model_name, system_prompt_type),
        load_personality("sarcasm", "prompting", model_name, system_prompt_type),
        load_personality("remorse", "prompting", model_name, system_prompt_type),
        load_personality("humor", "training", model_name, system_prompt_type),
        load_personality("sarcasm", "training", model_name, system_prompt_type),
        load_personality("remorse", "training", model_name, system_prompt_type),
    ]
    ds = concatenate_datasets(splits).shuffle(seed=123456)
    cols = [c for c in ds.column_names if c not in ["messages", "label"]]
    train_ds = ds.map(tokenize, remove_columns=cols)
    # validation dataset
    splits = [
        load_personality("humor", "prompting", model_name, system_prompt_type, split="test"),
        load_personality("sarcasm", "prompting", model_name, system_prompt_type, split="test"),
        load_personality("remorse", "prompting", model_name, system_prompt_type, split="test"),
        load_personality("humor", "training", model_name, system_prompt_type, split="test"),
        load_personality("sarcasm", "training", model_name, system_prompt_type, split="test"),
        load_personality("remorse", "training", model_name, system_prompt_type, split="test"),
    ]
    ds = concatenate_datasets(splits).shuffle(seed=123456)
    cols = [c for c in ds.column_names if c not in ["messages", "label"]]
    val_ds = ds.map(tokenize, remove_columns=cols)

    # train and save model
    collator = DataCollatorWithPadding(tokenizer)
    outpath = Path(f"{MODEL_PATH}/modernbert-base-classifier-exp")
    outpath.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=outpath,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        weight_decay=1e-6,
        num_train_epochs=5,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="modernbert-base-classifier",
        dataloader_num_workers=4,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=25,
    )
    metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels, average="macro")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(outpath)
    tokenizer.save_pretrained(outpath)

if __name__ == "__main__":
    train_classifier()