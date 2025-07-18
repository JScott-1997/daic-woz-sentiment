# src/train_bert.py
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer,
                          DataCollatorWithPadding, set_seed)
import numpy as np, evaluate, torch, pathlib

set_seed(42)
BASE = "distilbert-base-uncased"
TOK  = AutoTokenizer.from_pretrained(BASE)

data_file = "processed/bert/utterances_daic.jsonl"
ds = load_dataset("json", data_files=data_file, split="train")
ds = ds.train_test_split(test_size=0.2, stratify_by_column="label")
train_ds, val_ds = ds["train"], ds["test"]

def tokenize(b): return TOK(b["text"], truncation=True, max_length=128)
train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)
datacoll = DataCollatorWithPadding(tokenizer=TOK, return_tensors="pt")
metric   = evaluate.load("f1")

def compute_m(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)

args = TrainingArguments(
    output_dir="processed/bert/distilbert_daic",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(model, args, train_ds, val_ds,
                  tokenizer=TOK, data_collator=datacoll,
                  compute_metrics=compute_m)
trainer.train()

dest = pathlib.Path("processed/bert/distilbert_daic_best")
trainer.save_model(dest)
TOK.save_pretrained(dest)
print("Model saved →", dest)