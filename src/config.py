"""
config.py

Centralised paths & parameters.
Loads optional overrides from a `.env` file if python-dotenv is present,
but everything has a hard-coded fallback so the project works out-of-the-box.
"""

import os
from pathlib import Path

# ───── Optional .env support ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()  # silently does nothing if no .env file exists
except ImportError:
    # dotenv is optional; just continue with environment variables that
    # may already be set in the OS.
    pass

# ───── File-system paths ─────────────────────────────────────────────────
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
OUT_DIR = Path(os.getenv("OUT_DIR", "processed/transcripts_with_sentiment"))

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───── Sentiment → urgency mapping thresholds ───────────────────────────
NEG_HIGH = float(os.getenv("NEG_HIGH", "-0.6"))
NEG_MED = float(os.getenv("NEG_MED", "-0.2"))

# ───── (OPTIONAL) MongoDB settings – unused with stub ───────────────────
MONGO_URI = os.getenv("MONGO_URI", "")  # leave blank → no DB
DB_NAME = os.getenv("DB_NAME", "daic")
COLL_NAME = os.getenv("COLL_NAME", "utterances")
