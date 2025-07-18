from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env if present
load_dotenv()

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "processed" / "transcripts_with_sentiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Sentiment → urgency thresholds ----
NEG_HIGH = 0.95   # NEGATIVE & score ≥ this => HIGH urgency
NEG_MED  = 0.70   # NEGATIVE & score ≥ this => MEDIUM urgency

# ---- MongoDB ----
MONGO_URI = os.getenv("MONGO_URI")  # leave blank to skip DB
DB_NAME   = "mental_health"
COLL_NAME = "messages"
