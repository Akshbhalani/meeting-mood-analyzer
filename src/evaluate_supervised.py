# src/evaluate_supervised.py
import os
import sys
import json
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

def plot_confusion_matrix(cm, labels, outpath):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def build_pdf(metrics, cm_file, out_pdf):
    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Evaluation Report - Supervised Baseline", styles["Heading1"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(f"Accuracy: {metrics['accuracy']:.4f}", styles["Normal"]))
    rep_text = json.dumps(metrics["report"], indent=2)
    story.append(Paragraph("<pre>{}</pre>".format(rep_text.replace("<","&lt;").replace(">","&gt;")), styles["Code"]))
    story.append(Spacer(1,12))
    story.append(Image(cm_file, width=14*cm, height=8*cm))
    story.append(Spacer(1,12))
    doc.build(story)

def main(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")

    labels = ["positive","negative"]  # SST-2 is binary

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = clf.predict(X_test_tfidf)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist(), "labels": labels}

    # Save metrics
    with open(os.path.join(output_dir, "metrics_supervised.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    cm_path = os.path.join(output_dir, "confusion_supervised.png")
    plot_confusion_matrix(cm, labels, cm_path)

    # Save PDF
    pdf_path = os.path.join(output_dir, "evaluation_supervised.pdf")
    build_pdf(metrics, cm_path, pdf_path)

    print(f"✅ Metrics saved to {output_dir}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/evaluate_supervised.py data/test/sst2_sample.csv results/sst2_supervised")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
