import evaluate
import numpy as np
import pandas as pd
import torch as t
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, load_from_disk
from personality.constants import DATA_PATH, MODEL_PATH


data = load_from_disk(f"{DATA_PATH}/frontier-lmarena")
models = [
    'claude-3-opus-20240229',
    'claude-3-5-sonnet-20240620',
    'gpt-4o-2024-08-06',
    'gemini-1.5-pro-exp-0801',
    'deepseek-v2-api-0628',
    'llama-3.1-405b-instruct',
    'qwen2-72b-instruct',
    'mistral-large-2407'
]
data = data.filter(lambda x: x["model"] in models)



LABEL2ID = {m: i for i, m in enumerate(models)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

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
    messages = element["conversation"]
    prompt = messages[0]["content"]
    completion = messages[-1]["content"]
    text = f"Human: {prompt}\n\nAssistant: {completion}"
    out = tokenizer(text, truncation=True, max_length=8192)
    out["label"] = LABEL2ID[element["model"]]
    return out

train_test_split = data.train_test_split(test_size=0.2, seed=123456)
train_ds = train_test_split["train"]
test_ds = train_test_split["test"]

cols = [c for c in data.column_names if c not in ["label"]]
train_ds = train_ds.map(tokenize, remove_columns=cols)
val_ds = test_ds.map(tokenize, remove_columns=cols)

# train and save model
collator = DataCollatorWithPadding(tokenizer)
outpath = Path(f"{MODEL_PATH}/modernbert-base-classifier-frontier")
outpath.mkdir(parents=True, exist_ok=True)
train_args = TrainingArguments(
    output_dir=outpath,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    weight_decay=0.0,
    num_train_epochs=3,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=50,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="modernbert-base-classifier-frontier",
    dataloader_num_workers=4,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=50,
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