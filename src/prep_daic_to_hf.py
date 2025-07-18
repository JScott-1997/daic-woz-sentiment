"""
prep_daic_to_hf.py
────────────────────────────────────────────────────────────────────────────
• Build a JSON‑lines file with one record per utterance:
      {"text": "...", "label": 0/1}
  label = 1  ⇢  participant PHQ‑8 ≥ 10  (clinical depression)
  label = 0  ⇢  PHQ‑8 < 10

The script expects:
    processed/transcripts_with_sentiment/*_sentiment.csv
    data/labels/*_split_Depression_AVEC2017.csv
and writes:
    processed/bert/utterances_daic.jsonl
"""

from __future__ import annotations

import glob
import json  # noqa: F401  (kept for completeness if you want manual writes)
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ───────────────────────── paths ─────────────────────────────────────────
TRANS_DIR = Path("processed/transcripts_with_sentiment")
LABEL_DIR = Path("data/labels")
OUT_DIR = Path("processed/bert")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────── build session‑level label map ───────────────────────
digit_re = re.compile(r"\d+")
labels: dict[int, int] = {}

for lf in LABEL_DIR.glob("*_split_Depression_AVEC2017.csv"):
    if "test_split" in lf.name.lower():  # blind test set → skip
        continue
    df = pd.read_csv(lf, sep=None, engine="python")
    df.columns = [c.lower() for c in df.columns]  # standardise

    if not {"participant_id", "phq8_score"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {lf}: {df.columns}")

    for pid, score in zip(df["participant_id"], df["phq8_score"]):
        labels[int(pid)] = int(score >= 10)

print(f"Loaded PHQ‑8 labels for {len(labels)} participants")

# ─────────────────── iterate over sentiment CSVs ─────────────────────────
rows: list[dict[str, str | int]] = []

csv_files = sorted(TRANS_DIR.glob("*_sentiment.csv"))
if not csv_files:
    raise RuntimeError(f"No *_sentiment.csv files found in {TRANS_DIR}")

for csv_path in tqdm(csv_files, desc="Utterances"):
    m = digit_re.search(csv_path.name)
    if not m:
        continue  # filename without digits – should not happen
    session_id = int(m.group())
    label = labels.get(session_id)
    if label is None:
        # transcript without PHQ‑8 label (e.g., test split); skip
        continue

    df = pd.read_csv(csv_path)
    txt_col = next(
        (c for c in df.columns if c.lower() in {"utterance", "value", "text"}), None
    )
    if txt_col is None:
        raise KeyError(f"No utterance column in {csv_path}")

    for text in df[txt_col].astype(str):
        text = text.strip()
        if text:  # ignore empty lines
            rows.append({"text": text, "label": label})

# ─────────────────────────  write jsonl  ─────────────────────────────────
out_file = OUT_DIR / "utterances_daic.jsonl"
pd.DataFrame(rows).to_json(out_file, orient="records", lines=True)

print(f"Saved {len(rows):,} utterances → {out_file}")