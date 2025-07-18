# prep_daic_to_hf.py  – robust session‑split builder
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ───────── paths ────────────────────────────────────────────────────────
TRANS_DIR = Path("processed/transcripts_with_sentiment")
LABEL_DIR = Path("data/labels")            # holds *_split_Depression_AVEC2017.csv
OUT_DIR   = Path("processed/bert")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────── cli ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["official", "random", "strat"],
               default="official")
ap.add_argument("--train-frac", type=float, default=0.8)
ap.add_argument("--val-frac",   type=float, default=0.1)
ap.add_argument("--seed",       type=int,   default=42)
args = ap.parse_args()
random.seed(args.seed)

# ───────── load PHQ labels (headers made safe) ──────────────────────────
label_csvs = list(LABEL_DIR.glob("*_split_Depression_AVEC2017.csv"))
if not label_csvs:
    raise RuntimeError(f"No *_split_Depression_AVEC2017.csv found in {LABEL_DIR}")

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()          # tidy headers
    return df

phq_df = pd.concat(_clean(pd.read_csv(f)) for f in label_csvs)

# force numeric, NaN → -1
phq_df["phq8_score"] = pd.to_numeric(phq_df["phq8_score"], errors="coerce").fillna(-1)

phq_map = dict(zip(phq_df["participant_id"].astype(int), phq_df["phq8_score"].astype(int)))

print(f"Loaded PHQ table: {len(phq_map)} sessions "
      f"({sum(s>=10 for s in phq_map.values())} positives)")

# ───────── helper to extract session ID from filename ───────────────────
def sid_from_path(p: Path) -> int:
    # 300_TRANSCRIPT_sentiment.csv → 300
    return int(p.stem.split("_")[0])

# ───────── build utterance records ──────────────────────────────────────
records = []
for csv_f in tqdm(sorted(TRANS_DIR.glob("*_sentiment.csv")), desc="Utterances"):
    sid = sid_from_path(csv_f)
    df  = pd.read_csv(csv_f)

    col_text = "Utterance" if "Utterance" in df.columns else "value"
    for txt, sent in zip(df[col_text].astype(str), df["Sentiment"]):
        records.append({
            "text":   txt,
            "label":  1 if sent == "POSITIVE" else 0,
            "session": sid,
            "phq":    phq_map.get(sid, -1),
        })

all_sids = sorted({r["session"] for r in records})

# ───────── define train/val/test splits ─────────────────────────────────
def split_random(stratified: bool):
    pos = [s for s in all_sids if phq_map.get(s, -1) >= 10]
    neg = [s for s in all_sids if phq_map.get(s, -1) < 10]

    def _cut(lst):
        n_tr = int(len(lst) * args.train_frac)
        n_va = int(len(lst) * args.val_frac)
        return set(lst[:n_tr]), set(lst[n_tr:n_tr+n_va]), set(lst[n_tr+n_va:])

    if stratified:
        random.shuffle(pos); random.shuffle(neg)
        tr_p, va_p, te_p = _cut(pos)
        tr_n, va_n, te_n = _cut(neg)
        return tr_p|tr_n, va_p|va_n, te_p|te_n
    else:
        both = pos + neg
        random.shuffle(both)
        return _cut(both)

if args.split == "official":
    train_ids = set(pd.read_csv(LABEL_DIR/"train_split_Depression_AVEC2017.csv")["Participant_ID"])
    val_ids   = set(pd.read_csv(LABEL_DIR/"dev_split_Depression_AVEC2017.csv")["Participant_ID"])
    test_ids  = set(pd.read_csv(LABEL_DIR/"test_split_Depression_AVEC2017.csv")["Participant_ID"])
else:
    train_ids, val_ids, test_ids = split_random(args.split == "strat")

splits = {"train": train_ids, "val": val_ids, "test": test_ids}

# ───────── dump JSONL files ─────────────────────────────────────────────
for name, ids in splits.items():
    out_f = OUT_DIR / f"{name}.jsonl"
    with out_f.open("w", encoding="utf-8") as fp:
        for r in records:
            if r["session"] in ids:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name:<5} {len(ids):3} sessions → {out_f}")
