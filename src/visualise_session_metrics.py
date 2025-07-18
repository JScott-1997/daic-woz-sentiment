# ──────────────────  src/visualise_session_metrics.py  ──────────────────
import argparse, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def main(thr: float = 0.33, save: bool = False):
    csv_f = pathlib.Path("processed/bert/session_scores.csv")
    if not csv_f.exists():
        raise FileNotFoundError(f"{csv_f} not found – run session_infer.py first")

    df = pd.read_csv(csv_f)
    df = df.dropna(subset=["label"]).astype({"label": int})
    y_true, y_score = df["label"].values, df["p_dep"].values
    y_pred = (y_score >= thr).astype(int)

    # ‑‑ 1) Histogram
    plt.figure()
    bins = 20
    plt.hist(y_score[y_true == 0], bins=bins, alpha=0.6, label="PHQ < 10")
    plt.hist(y_score[y_true == 1], bins=bins, alpha=0.6, label="PHQ ≥ 10")
    plt.axvline(thr, linestyle="--", color="k")
    plt.xlabel("Predicted P(depressed)")
    plt.ylabel("Number of sessions")
    plt.title("Distribution of session‑level probabilities")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("processed/bert/hist_probs.png", dpi=300)
    else:
        plt.show()

    # ‑‑ 2) ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.xlabel("False‑positive rate")
    plt.ylabel("True‑positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("processed/bert/roc_curve.png", dpi=300)
    else:
        plt.show()

    # ‑‑ 3) Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["PHQ<10", "PHQ≥10"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"Confusion matrix  (thr={thr})")
    plt.tight_layout()
    if save:
        plt.savefig("processed/bert/conf_matrix.png", dpi=300)
    else:
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.33, help="decision threshold")
    ap.add_argument("--save", action="store_true", help="save PNGs instead of showing")
    args = ap.parse_args()
    main(thr=args.thr, save=args.save)
