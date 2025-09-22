# src/evaluate_text_baselines.py
import os
import sys
import json
import argparse
from io import BytesIO

import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# ---------- helpers ----------
def predict_textblob(text, pos_thr=0.1, neg_thr=-0.1):
    tb = TextBlob(text or "")
    pol = tb.sentiment.polarity
    if pol > pos_thr:
        return "positive", pol
    if pol < neg_thr:
        return "negative", pol
    return "neutral", pol

def predict_vader(text, pos_thr=0.05, neg_thr=-0.05, sia=None):
    if sia is None:
        sia = SentimentIntensityAnalyzer()
    sc = sia.polarity_scores(text or "")["compound"]
    if sc >= pos_thr:
        return "positive", sc
    if sc <= neg_thr:
        return "negative", sc
    return "neutral", sc

def compute_metrics(y_true, y_pred, labels=["positive","neutral","negative"]):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist(), "labels": labels}

def plot_confusion_matrix(cm, labels, outpath):
    plt.figure(figsize=(6,4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def build_pdf(metrics_all, cm_files, out_pdf):
    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Evaluation Report - Text Baselines", styles["Heading1"]))
    story.append(Spacer(1,12))
    for method, metrics in metrics_all.items():
        story.append(Paragraph(f"<b>Method:</b> {method}", styles["Heading2"]))
        story.append(Paragraph(f"Accuracy: {metrics['accuracy']:.4f}", styles["Normal"]))
        # add classification report text
        rep_text = json.dumps(metrics["report"], indent=2)
        story.append(Paragraph("<pre>{}</pre>".format(rep_text.replace("<","&lt;").replace(">","&gt;")), styles["Code"]))
        story.append(Spacer(1,12))
        # attach confusion matrix image if exists
        if method in cm_files:
            story.append(Image(cm_files[method], width=14*cm, height=8*cm))
            story.append(Spacer(1,12))
    doc.build(story)

# ---------- main ----------
def main(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")

    labels = ["positive","neutral","negative"]
    sia = SentimentIntensityAnalyzer()

    preds = {"textblob": [], "vader": []}
    scores = {"textblob": [], "vader": []}

    for text in df["text"].astype(str).tolist():
        tb_lab, tb_score = predict_textblob(text)
        va_lab, va_score = predict_vader(text, sia=sia)
        preds["textblob"].append(tb_lab)
        preds["vader"].append(va_lab)
        scores["textblob"].append(tb_score)
        scores["vader"].append(va_score)

    df["pred_textblob"] = preds["textblob"]
    df["score_textblob"] = scores["textblob"]
    df["pred_vader"] = preds["vader"]
    df["score_vader"] = scores["vader"]

    # Save predictions
    df.to_csv(os.path.join(output_dir, "predictions_with_scores.csv"), index=False)

    # Compute metrics
    metrics_all = {}
    cm_files = {}
    for method in ["textblob","vader"]:
        y_true = df["label"].astype(str).tolist()
        y_pred = df[f"pred_{method}"].astype(str).tolist()
        metrics = compute_metrics(y_true, y_pred, labels=labels)
        metrics_all[method] = metrics
        # confusion matrix plot
        cm_path = os.path.join(output_dir, f"confusion_{method}.png")
        plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], cm_path)
        cm_files[method] = cm_path

    # Save metrics json
    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)

    # Build PDF report
    pdf_path = os.path.join(output_dir, "evaluation_report.pdf")
    try:
        build_pdf(metrics_all, cm_files, pdf_path)
        print("PDF report saved to:", pdf_path)
    except Exception as e:
        print("Failed to build PDF (reportlab):", e)

    print("Metrics summary saved to:", os.path.join(output_dir, "metrics_summary.json"))
    print("Predictions saved to:", os.path.join(output_dir, "predictions_with_scores.csv"))
    for method in metrics_all:
        print(f"Method: {method} Accuracy: {metrics_all[method]['accuracy']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/evaluate_text_baselines.py data/test/sentiment_test.csv results/eval")
        sys.exit(1)
    input_csv = sys.argv[1]
    outdir = sys.argv[2]
    main(input_csv, outdir)
