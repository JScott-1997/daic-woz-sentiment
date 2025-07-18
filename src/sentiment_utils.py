from transformers import pipeline
from typing import Tuple

# Lazy‑load so it happens once
_sentiment_pipe = None

def load_pipeline():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline("sentiment-analysis")
    return _sentiment_pipe


def analyse_message(text: str) -> Tuple[str, float]:
    """
    Returns (label, score) from Hugging Face sentiment pipeline.
    """
    pipe = load_pipeline()
    result = pipe(text, truncation=True)[0]
    return result["label"], float(result["score"])


def urgency_from_sentiment(label: str, score: float,
                            neg_high: float, neg_med: float) -> str:
    """
    Maps sentiment → LOW | MEDIUM | HIGH urgency.
    """
    if label == "NEGATIVE" and score >= neg_high:
        return "HIGH"
    if label == "NEGATIVE" and score >= neg_med:
        return "MEDIUM"
    return "LOW"