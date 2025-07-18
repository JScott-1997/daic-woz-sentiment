# src/prep_daic_to_hf.py
import pandas as pd, glob, re, json, pathlib
from tqdm import tqdm

DIGIT = re.compile(r"\d+")
rows   = []

# --- 1.  map Session → binary PHQ label -----------------------
labels = {}
for lf in glob.glob("data/labels/*_split_Depression_AVEC2017.csv"):
    if "test_split" in lf.lower():        # blind set, no PHQ
        continue
    df = pd.read_csv(lf, sep=None, engine="python")
    df.columns = [c.lower() for c in df.columns]
    for sid, score in zip(df["participant_id"], df["phq8_score"]):
        labels[int(sid)] = int(score >= 10)

# --- 2.  iterate over your existing sentiment CSVs ------------
for f in tqdm(glob.glob("processed/transcripts_with_sentiment/*_sentiment.csv")):
    sid = int(DIGIT.search(pathlib.Path(f).name).group())
    label = labels.get(sid)
    if label is None:
        continue                           # skip if no PHQ
    df = pd.read_csv(f)
    txt_col = next(c for c in df.columns if c.lower() in {"utterance","value"})
    for text in df[txt_col].astype(str):
        rows.append({"text": text.strip(), "label": label})

out = pathlib.Path("processed/bert")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_json(out / "utterances_daic.jsonl",
                           orient="records", lines=True)
print("Saved", len(rows), "utterances →", out / "utterances_daic.jsonl")