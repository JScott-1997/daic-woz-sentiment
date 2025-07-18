"""
batch_daic.py
────────────────────────────────────────────────────────────────────────────
Process all *_P.zip files in data/raw:

  • Extract *_TRANSCRIPT.csv
  • Robustly read (comma OR TAB, UTF‑8 OR latin‑1)
  • Run DistilBERT sentiment analysis on each utterance
  • Map signed score → Urgency (LOW / MEDIUM / HIGH)
  • Save annotated CSVs to processed/transcripts_with_sentiment/

MongoDB insert has been disabled; insert_doc is a no‑op stub so the script
has no external dependencies beyond the Python packages in requirements.txt.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from config import RAW_DIR, OUT_DIR, NEG_HIGH, NEG_MED

# ─────────────────── Mongo stub (no‑op) ──────────────────────────────────
def insert_doc(*_, **__) -> None:  # kept so other modules can import safely
    return


# ─────────────────── Sentiment pipeline (GPU if available) ───────────────
try:
    import torch

    DEVICE = 0 if torch.cuda.is_available() else -1
except ImportError:
    DEVICE = -1

sent_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=DEVICE,
    batch_size=32,
)


# ──────────────────────  helpers  ────────────────────────────────────────
def urgency_from_sent(score: float) -> str:
    """
    Map a signed sentiment score to an urgency bucket.
    (Negative score => negative sentiment.)
    """
    if score < NEG_HIGH:
        return "HIGH"
    if score < NEG_MED:
        return "MEDIUM"
    return "LOW"


def _read_csv_bytes(raw: bytes, *, sep: str | None = None) -> pd.DataFrame:
    """
    Read CSV bytes, trying UTF‑8 first and falling back to latin‑1 so that
    stray non‑UTF bytes don't crash the parser.
    """
    try:
        return pd.read_csv(io.BytesIO(raw), sep=sep)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), sep=sep, encoding="latin-1")


def process_csv_bytes(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Return a DataFrame with Sentiment / Confidence / Urgency columns."""

    # 1️⃣ first attempt: assume comma‑separated
    df = _read_csv_bytes(raw_bytes)

    # 2️⃣ if only one weird column, reload as TAB‑sep
    if len(df.columns) == 1 and "\t" in df.columns[0]:
        df = _read_csv_bytes(raw_bytes, sep="\t")

    # Identify the utterance/text column
    text_col = next(
        (c for c in df.columns if c.lower() in {"utterance", "value", "text"}),
        None,
    )
    if text_col is None:
        raise KeyError(
            f"No text column found in {filename}; columns={list(df.columns)}"
        )

    # ── Sentiment inference ──────────────────────────────────────────────
    texts = df[text_col].astype(str).tolist()
    results = sent_pipe(texts, truncation=True)

    sentiments = [
        "NEGATIVE" if r["label"] == "NEGATIVE" else "POSITIVE" for r in results
    ]
    signed_scores = [
        -r["score"] if r["label"] == "NEGATIVE" else r["score"] for r in results
    ]
    urgencies = [urgency_from_sent(s) for s in signed_scores]

    df["Sentiment"] = sentiments
    df["Confidence"] = [abs(s) for s in signed_scores]
    df["Urgency"] = urgencies
    return df


# ────────────────────  main  ─────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        print(f"RAW_DIR {RAW_DIR} does not exist; nothing to process.")
        return

    zip_paths = sorted(RAW_DIR.glob("*_P.zip"))
    if not zip_paths:
        print(f"No *_P.zip files found in {RAW_DIR}")
        return

    for zp in tqdm(zip_paths, desc="📦 Processing"):
        with zipfile.ZipFile(zp) as zf:
            # locate the transcript inside the zip
            csv_info = next(
                (i for i in zf.infolist() if i.filename.endswith("_TRANSCRIPT.csv")),
                None,
            )
            if not csv_info:
                print(f"   ⚠ No transcript CSV found in {zp.name}")
                continue

            try:
                df = process_csv_bytes(zf.read(csv_info.filename), csv_info.filename)
            except Exception as exc:
                print(f"   ⚠ Error processing {csv_info.filename}: {exc}")
                continue

            out_name = OUT_DIR / f"{csv_info.filename[:-4]}_sentiment.csv"
            df.to_csv(out_name, index=False)
            print(f"   ✔ Saved {out_name}")

            # optional DB hook – currently no‑op
            # session_id = int(zp.stem.split('_')[0])
            # insert_doc(session_id, df)

    print("🎉  All done!")


if __name__ == "__main__":
    main()
