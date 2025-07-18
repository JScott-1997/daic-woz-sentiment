# src/mkrir.py
from pathlib import Path

# root = project folder where this script lives
root = Path(__file__).resolve().parents[1]

# folders your pipeline needs
folders = [
    root / "processed",
    root / "processed" / "transcripts_with_sentiment",
    root / "processed" / "bert",
    root / "data" / "labels",
]

for p in folders:
    p.mkdir(parents=True, exist_ok=True)
    print("Ensured →", p)