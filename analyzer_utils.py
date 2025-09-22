import re
import io
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# PDF report
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# Download NLTK resources if not already installed
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize VADER
vader = SentimentIntensityAnalyzer()


# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_text(text: str) -> str:
    """Clean and preprocess input text using regex, NLTK stopwords, and spaCy lemmatization."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation & digits
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)


# ----------------------------
# Sentiment Analysis
# ----------------------------
def analyze_textblob(text: str) -> dict:
    """Analyze sentiment using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return {
        "method": "TextBlob",
        "polarity": polarity,
        "subjectivity": subjectivity,
        "label": "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral",
    }


def analyze_vader(text: str) -> dict:
    """Analyze sentiment using VADER."""
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
    return {
        "method": "VADER",
        "compound": compound,
        "label": label,
    }


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(y_true, y_pred, model_name="Model"):
    """Compute evaluation metrics for classification results."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=["positive", "neutral", "negative"])

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # Plot confusion matrix
    buf = io.BytesIO()
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["positive", "neutral", "negative"],
        yticklabels=["positive", "neutral", "negative"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return metrics, buf


# ----------------------------
# Visualizations
# ----------------------------
def generate_wordcloud(text: str, title="Word Cloud"):
    """Generate a word cloud from text and return buffer."""
    buf = io.BytesIO()
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf


# ----------------------------
# PDF Export
# ----------------------------
def save_pdf(results: dict, charts: dict, filename="meeting_analysis_report.pdf", return_bytes=False):
    """
    Generate a PDF report with metrics and charts.
    results: dictionary with metrics & info
    charts: dictionary of matplotlib chart buffers {name: BytesIO}
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph("<b>Meeting Mood Analyzer - Text Report</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Date
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {date_str}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Metrics Table
    if "metrics" in results:
        data = [["Model", "Accuracy", "Precision", "Recall", "F1-Score"]]
        for metric in results["metrics"]:
            data.append([
                metric["model"],
                f"{metric['accuracy']:.3f}",
                f"{metric['precision']:.3f}",
                f"{metric['recall']:.3f}",
                f"{metric['f1_score']:.3f}",
            ])
        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), "#d3d3d3"),
            ("GRID", (0, 0), (-1, -1), 1, "black"),
        ]))
        elements.append(Paragraph("<b>Evaluation Metrics</b>", styles["Heading2"]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    # Insert charts
    for name, buf in charts.items():
        buf.seek(0)
        img = Image(buf, width=400, height=250)
        elements.append(Paragraph(f"<b>{name}</b>", styles["Heading2"]))
        elements.append(img)
        elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)

    if return_bytes:
        pdf = buffer.getvalue()
        buffer.close()
        return pdf
    else:
        with open(filename, "wb") as f:
            f.write(buffer.getvalue())
        buffer.close()
        return filename

