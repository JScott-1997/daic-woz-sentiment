"""
train_bert.py
────────────────────────────────────────────────────────────────────────────
Fine‑tune DistilBERT on utterance‑level depression labels.

Prerequisites
-------------
• processed/bert/utterances_daic.jsonl  (built by prep_daic_to_hf.py)
• pip install evaluate datasets accelerate transformers torch

Outputs
-------
• processed/bert/distilbert_daic_best/  (HF model & tokenizer files)
• console log with train / eval metrics every epoch
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import evaluate
import numpy as np
import torch
from datasets import ClassLabel, Dataset, load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

# ───────────────────────── paths ─────────────────────────────────────────
DATA_F = pathlib.Path("processed/bert/utterances_daic.jsonl")
OUT_DIR = pathlib.Path("processed/bert/distilbert_daic_best")
MODEL_NAME = "distilbert-base-uncased"

# ─────────────────── 1. load dataset  ────────────────────────────────────
ds: Dataset = load_dataset("json", data_files=str(DATA_F), split="train")
# Convert int → ClassLabel so stratify works
ds = ds.cast_column("label", ClassLabel(names=["neg", "pos"]))

# Stratified 80/20 split
ds = ds.train_test_split(test_size=0.2, stratify_by_column="label")
train_ds, val_ds = ds["train"], ds["test"]

print(train_ds)
print(val_ds)

# ─────────────────── 2. tokenizer & tokenisation  ────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tok(batch["text"], truncation=True)


train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tok)

# ─────────────────── 3. model  ───────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# ─────────────────── 4. metrics  ─────────────────────────────────────────
metric_f1 = evaluate.load("f1")
metric_acc = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)[
            "accuracy"
        ],
        "f1": metric_f1.compute(
            predictions=preds, references=labels, average="weighted"
        )["f1"],
    }


# ─────────────────── 5. training args  ───────────────────────────────────
training_args = TrainingArguments(
    output_dir="processed/bert/runs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # mixed precision on GPU
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
)

# ─────────────────── 6. train  ───────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ─────────────────── 7. save best  ───────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"Best checkpoint saved → {OUT_DIR}")
