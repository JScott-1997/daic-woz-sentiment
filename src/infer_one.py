# scripts/snippets/infer_one.py  (ad‑hoc)
import torch, pathlib, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = pathlib.Path("processed/bert/distilbert_daic_best")
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def p_depressed(text: str) -> float:
    t = tok(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**t).logits
    return float(logits.softmax(-1)[0, 1])   # class 1 = PHQ≥10

for sent in [
    "I feel hopeless about everything.",
    "Things are actually going pretty well lately.",
    "I can't get out of bed most mornings.",
]:
    print(f"{sent!r:50s}  →  P(PHQ≥10) = {p_depressed(sent):.2f}")