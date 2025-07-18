"""
session_infer.py
────────────────────────────────────────────────────────────────────────────
Aggregate utterance‑level probabilities into session‑level scores and
compare with PHQ‑8 ground truth.

* Reads fine‑tuned checkpoint at processed/bert/distilbert_daic_best/
* Uses mean P(depressed) across all utterances in a session
* Default threshold 0.33  (override with --thr 0.5)
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ────────────────────────── CLI ──────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--thr", type=float, default=0.33, help="decision threshold")
args = ap.parse_args()

THR = args.thr

# ────────────────────────── paths ────────────────────────────────────────
MODEL_DIR = Path("processed/bert/distilbert_daic_best")
TRANS_DIR = Path("processed/transcripts_with_sentiment")
LABEL_DIR = Path("data/labels")

# ─────────────────── 1. load fine‑tuned model  ───────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
DEVICE = 0 if torch.cuda.is_available() else -1

# ─────────────────── 2. build session → mean‑prob  ───────────────────────
digit_re = re.compile(r"\d+")
records: list[dict] = []

for csv_f in tqdm(sorted(TRANS_DIR.glob("*_sentiment.csv")), desc="Sessions"):
    sid = int(digit_re.search(csv_f.name).group())
    df = pd.read_csv(csv_f)

    # If we saved prob earlier you can load it directly.
    if "p_depressed" in df.columns:
        probs = df["p_depressed"].values
    else:
        # Compute with model (rarely needed after batch_daic enhancement)
        texts = df[next(c for c in df.columns if c.lower() in {"utterance","value","text"})].astype(str).tolist()
        # batch inference
        probs = []
        for i in range(0, len(texts), 32):
            inputs = tok(texts[i : i + 32], return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = model(**{k: v.to(model.device) for k, v in inputs.items()}).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())

    records.append(
        {
            "session": sid,
            "p_dep": float(np.mean(probs)),
        }
    )

score_df = pd.DataFrame(records)

# ─────────────────── 3. merge PHQ‑8 labels  ──────────────────────────────
labels = {}
for lf in LABEL_DIR.glob("*_split_Depression_AVEC2017.csv"):
    if "test_split" in lf.name.lower():
        continue
    df_lab = pd.read_csv(lf, sep=None, engine="python")
    df_lab.columns = [c.lower() for c in df_lab.columns]
    phq_col = next(c for c in df_lab.columns if "phq" in c and "score" in c)
    for pid, score in zip(df_lab["participant_id"], df_lab[phq_col]):
        labels[int(pid)] = int(score >= 10)

score_df["label"] = score_df["session"].map(labels)

# ─────────────────── 4. drop sessions without label  ─────────────────────
score_df = score_df.dropna(subset=["label"])
score_df["label"] = score_df["label"].astype(int)

print(
    f"\nAnalysed {len(score_df)} sessions  "
    f"(positives={ (score_df['label']==1).sum() }, "
    f"negatives={ (score_df['label']==0).sum() })  (threshold={THR})\n"
)

# ─────────────────── 5. apply threshold & metrics  ───────────────────────
score_df["pred"] = (score_df["p_dep"] >= THR).astype(int)

# save once for visualisations
Path("processed/bert").mkdir(parents=True, exist_ok=True)
score_df.to_csv("processed/bert/session_scores.csv", index=False)
print("Wrote  processed/bert/session_scores.csv")

print("Confusion matrix:")
print(confusion_matrix(score_df["label"], score_df["pred"]), "\n")

print("Classification report:")
print(
    classification_report(score_df["label"], score_df["pred"], digits=3)
)
