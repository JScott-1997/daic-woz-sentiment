# src/session_infer.py
import pandas as pd, glob, re, torch, pathlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = "processed/bert/distilbert_daic_best"
tok    = AutoTokenizer.from_pretrained(ckpt)
model  = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
model.eval()

DIGIT = re.compile(r"\d+")
sess_rows = []

for f in tqdm(glob.glob("processed/transcripts_with_sentiment/*_sentiment.csv")):
    sid = int(DIGIT.search(pathlib.Path(f).name).group())
    df  = pd.read_csv(f)
    txt_col = next(c for c in df.columns if c.lower() in {"utterance","value"})
    texts = df[txt_col].astype(str).tolist()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i+32], truncation=True, padding=True,
                      max_length=128, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs.extend(torch.softmax(logits, dim=1)[:,1].cpu().tolist())
    sess_rows.append({"Session": sid, "mean_prob": sum(probs)/len(probs)})

score_df = pd.DataFrame(sess_rows)

# merge PHQ labels
labs = {}
for lf in glob.glob("data/labels/*_split_Depression_AVEC2017.csv"):
    if "test_split" in lf.lower(): continue
    df = pd.read_csv(lf); df.columns = [c.lower() for c in df.columns]
    for sid, sc in zip(df["participant_id"], df["phq8_score"]):
        labs[int(sid)] = int(sc >= 10)
score_df["label"] = score_df["Session"].map(labs)

thr = 0.33
score_df["pred"] = (score_df["mean_prob"] >= thr).astype(int)

print("Confusion matrix (thr=0.33):")
print(confusion_matrix(score_df["label"], score_df["pred"]), "\n")
print(classification_report(score_df["label"], score_df["pred"],
                            target_names=["PHQ<10","PHQ>=10"], zero_division=0))
