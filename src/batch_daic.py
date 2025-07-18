import zipfile, csv, io, re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sentiment_utils import analyse_message, urgency_from_sentiment
from config import RAW_DIR, OUT_DIR, NEG_HIGH, NEG_MED
from mongo_utils import insert_doc

TRANSCRIPT_PATTERN = re.compile(r"_TRANSCRIPT\.csv$", re.I)  # matches “…_TRANSCRIPT.csv”

def process_transcript(file_bytes: bytes, file_name: str):
    """
    Takes CSV bytes from inside ZIP, returns a DataFrame with sentiment cols.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Speaker" in df.columns:
        df = df[df["Speaker"] == "Participant"]

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=file_name, leave=False):
        text = str(row["Utterance"])
        label, score = analyse_message(text)
        urgency = urgency_from_sentiment(label, score, NEG_HIGH, NEG_MED)
        results.append({**row,
                        "Sentiment": label,
                        "Confidence": round(score, 3),
                        "Urgency": urgency})
        # Optional DB insert (comment out if not needed)
        insert_doc({
            "file": file_name,
            "participant": row.get("Participant_ID", "unknown"),
            "message": text,
            "sentiment": label,
            "confidence": score,
            "urgency": urgency,
        })

    return pd.DataFrame(results)


def main():
    for zip_path in RAW_DIR.glob("*.zip"):
        print(f"📦 Processing {zip_path.name}")
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if TRANSCRIPT_PATTERN.search(info.filename):
                    file_bytes = z.read(info.filename)
                    df_out = process_transcript(file_bytes, info.filename)
                    out_name = Path(info.filename).stem + "_sentiment.csv"
                    out_file = OUT_DIR / out_name
                    df_out.to_csv(out_file, index=False)
                    print(f"   ✔ Saved {out_file.relative_to(OUT_DIR.parent)}")

    print("🎉 All done!")

if __name__ == "__main__":
    main()