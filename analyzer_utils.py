# import os
# import json
# import uuid
# from datetime import datetime
# from io import BytesIO
# from typing import Dict, List, Optional, Tuple

# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from reportlab.lib.pagesizes import A4
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib.units import cm

# # Sentiment + NLP
# from textblob import TextBlob
# from transformers import pipeline

# # Audio/Video
# import speech_recognition as sr
# from moviepy.editor import VideoFileClip

# # --------------------------
# # Paths & constants
# # --------------------------
# SESSIONS_DIR = os.path.join("results", "sessions")
# os.makedirs(SESSIONS_DIR, exist_ok=True)

# DEFAULT_EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]

# # Load HuggingFace pipeline lazily
# try:
#     hf_sentiment = pipeline("sentiment-analysis")
# except Exception:
#     hf_sentiment = None

# # --------------------------
# # Hybrid Sentiment Analysis
# # --------------------------
# def _hybrid_sentiment(text: str) -> Tuple[str, float, float, float]:
#     """
#     Combine TextBlob + HuggingFace for sentiment.
#     Returns (label, score, polarity, subjectivity).
#     """
#     if not text.strip():
#         return "Neutral", 0.0, 0.0, 0.0

#     # TextBlob
#     tb = TextBlob(text)
#     polarity = float(tb.sentiment.polarity)
#     subjectivity = float(tb.sentiment.subjectivity)

#     # HuggingFace (fallback if not available)
#     label = None
#     score = 0.0
#     if hf_sentiment:
#         try:
#             res = hf_sentiment(text[:512])[0]  # truncate long text
#             label = res.get("label", None)
#             score = float(res.get("score", 0.0))
#         except Exception:
#             label = None

#     # Harmonize label
#     if label:
#         label_lower = label.lower()
#         if label_lower in ["pos", "positive", "label_1"]:
#             label = "Positive"
#         elif label_lower in ["neg", "negative", "label_0"]:
#             label = "Negative"
#         else:
#             label = None  # fallback to TextBlob
#     if not label:
#         # Fallback: use polarity
#         if polarity > 0.1:
#             label = "Positive"
#         elif polarity < -0.1:
#             label = "Negative"
#         else:
#             label = "Neutral"

#     return label, score, polarity, subjectivity

# # --------------------------
# # Word Frequency
# # --------------------------
# def _simple_word_freq(text: str, top_n: int = 30) -> Dict[str, int]:
#     stop = set("""
#         the a an and or is are was were be been being i me my we our you your he she it
#         they them this that to of in for on with as at by from up down out over under about
#         not no yes do did done have has had can could would should may might will just
#         if then there here when where who what why how
#     """.split())
#     tokens = []
#     for raw in text.lower().replace("\n", " ").split():
#         tok = "".join(ch for ch in raw if ch.isalnum())
#         if tok and tok not in stop and len(tok) > 2:
#             tokens.append(tok)
#     freq = {}
#     for t in tokens:
#         freq[t] = freq.get(t, 0) + 1
#     return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

# # --------------------------
# # Core analyzers
# # --------------------------
# def analyze_text(text: str) -> Dict:
#     label, score, polarity, subjectivity = _hybrid_sentiment(text or "")
#     wf = _simple_word_freq(text or "")

#     # Tiny heuristic emotion map
#     if polarity >= 0.1:
#         emotions = {"happy": 0.6, "surprise": 0.15, "neutral": 0.25}
#     elif polarity <= -0.1:
#         emotions = {"sad": 0.5, "angry": 0.2, "neutral": 0.3}
#     else:
#         emotions = {"neutral": 0.7, "happy": 0.15, "sad": 0.15}

#     return {
#         "sentiment": label,
#         "score": score,
#         "polarity": polarity,
#         "subjectivity": subjectivity,
#         "word_freq": wf,
#         "emotions": emotions,
#         "transcript": text,
#     }


# def analyze_audio(audio_file) -> Dict:
#     recognizer = sr.Recognizer()
#     text = None
#     try:
#         with sr.AudioFile(audio_file) as source:
#             audio = recognizer.record(source)
#             text = recognizer.recognize_google(audio)
#     except Exception:
#         text = "Simulated transcript: " \
#                "audio could not be processed, using TextBlob fallback sentiment."

#     return analyze_text(text)


# def analyze_video(video_file) -> Dict:
#     transcript = None
#     try:
#         clip = VideoFileClip(video_file)
#         audio_path = video_file + "_temp.wav"
#         clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
#         transcript = analyze_audio(audio_path)["transcript"]
#         os.remove(audio_path)
#     except Exception:
#         transcript = "Simulated transcript: video could not be processed, using TextBlob fallback sentiment."

#     return analyze_text(transcript)

# # --------------------------
# # Persistence & history
# # --------------------------
# def save_session(results: Dict, directory: str = SESSIONS_DIR) -> str:
#     os.makedirs(directory, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     uid = uuid.uuid4().hex[:8]
#     path = os.path.join(directory, f"session_{ts}_{uid}.json")
#     payload = {
#         "timestamp": datetime.now().isoformat(timespec="seconds"),
#         **results
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)
#     return path


# def load_sessions(directory: str = SESSIONS_DIR) -> pd.DataFrame:
#     if not os.path.isdir(directory):
#         return pd.DataFrame()
#     rows = []
#     for name in sorted(os.listdir(directory)):
#         if not name.endswith(".json"):
#             continue
#         try:
#             with open(os.path.join(directory, name), "r", encoding="utf-8") as f:
#                 rows.append(json.load(f))
#         except Exception:
#             continue
#     if not rows:
#         return pd.DataFrame()
#     df = pd.DataFrame(rows)
#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#         df = df.sort_values("timestamp")
#     if "emotions" in df.columns:
#         emo_df = df["emotions"].apply(lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series())
#         emo_df = emo_df.fillna(0.0)
#         for col in emo_df.columns:
#             df[f"emo_{col}"] = emo_df[col]
#     return df

# # --------------------------
# # Plot helpers
# # --------------------------
# def _save_current_fig_to_buf(dpi: int = 150) -> BytesIO:
#     buf = BytesIO()
#     plt.tight_layout()
#     plt.savefig(buf, format="PNG", dpi=dpi, bbox_inches="tight")
#     plt.close()
#     buf.seek(0)
#     return buf

# def plot_wordcloud(word_freq: Dict[str, int]) -> BytesIO:
#     wc = WordCloud(width=900, height=450, background_color="white")
#     if word_freq:
#         wc = wc.generate_from_frequencies(word_freq)
#         img = wc.to_image()
#         buf = BytesIO()
#         img.save(buf, format="PNG")
#         buf.seek(0)
#         return buf
#     plt.figure(figsize=(6, 3))
#     plt.text(0.5, 0.5, "No words", ha="center", va="center")
#     return _save_current_fig_to_buf()

# def plot_word_freq(word_freq: Dict[str, int], top_n: int = 20) -> BytesIO:
#     items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
#     labels = [k for k, _ in items]
#     values = [v for _, v in items]
#     plt.figure(figsize=(7, 4))
#     plt.bar(labels, values)
#     plt.xticks(rotation=45, ha="right")
#     plt.title("Word Frequency")
#     plt.ylabel("Count")
#     return _save_current_fig_to_buf()

# def plot_emotions_bar(emotions: Dict[str, float], show: Optional[List[str]] = None) -> BytesIO:
#     emo = emotions or {}
#     if show:
#         emo = {k: v for k, v in emo.items() if k in show}
#     labels = list(emo.keys())
#     values = list(emo.values())
#     plt.figure(figsize=(6, 3.6))
#     plt.bar(labels, values)
#     plt.ylim(0, 1)
#     plt.title("Emotions (Bar)")
#     return _save_current_fig_to_buf()

# def plot_emotion_radar(emotions: Dict[str, float], show: Optional[List[str]] = None) -> BytesIO:
#     emo = emotions or {}
#     if show:
#         emo = {k: v for k, v in emo.items() if k in show}
#     if not emo:
#         plt.figure(figsize=(4, 4))
#         plt.text(0.5, 0.5, "No emotions", ha="center", va="center")
#         return _save_current_fig_to_buf()

#     labels = list(emo.keys())
#     values = list(emo.values())
#     angles = [n / float(len(labels)) * 2.0 * 3.14159265 for n in range(len(labels))]
#     values = values + values[:1]
#     angles = angles + angles[:1]

#     plt.figure(figsize=(4.5, 4.5))
#     ax = plt.subplot(111, polar=True)
#     ax.plot(angles, values)
#     ax.fill(angles, values, alpha=0.25)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels([])
#     ax.set_title("Emotions (Radar)")
#     return _save_current_fig_to_buf()

# def plot_metric_trend(df: pd.DataFrame, metric: str) -> BytesIO:
#     if df.empty or metric not in df.columns:
#         plt.figure(figsize=(6, 3))
#         plt.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
#         return _save_current_fig_to_buf()
#     plt.figure(figsize=(7, 3.8))
#     plt.plot(df["timestamp"], df[metric], marker="o")
#     plt.title(f"Trend: {metric}")
#     plt.xlabel("Time")
#     plt.ylabel(metric)
#     plt.xticks(rotation=30, ha="right")
#     return _save_current_fig_to_buf()

# def plot_emotion_trend(df: pd.DataFrame, emotion: str) -> BytesIO:
#     col = f"emo_{emotion}"
#     return plot_metric_trend(df, col)

# # --------------------------
# # Export helpers
# # --------------------------
# def save_json(results: Dict, json_path: str) -> None:
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

# def save_csv(results: Dict, csv_path: str) -> None:
#     df = pd.DataFrame([results])
#     df.to_csv(csv_path, index=False)

# def _dominant_emotion(emotions: Dict[str, float]) -> Optional[Tuple[str, float]]:
#     if not emotions:
#         return None
#     k = max(emotions, key=lambda x: emotions[x])
#     return k, emotions[k]

# def _auto_conclusion(results: Dict, history: Optional[pd.DataFrame] = None) -> str:
#     lines = []
#     sent = results.get("sentiment", "Unknown")
#     pol = results.get("polarity", 0.0)
#     dom = _dominant_emotion(results.get("emotions", {}))
#     lines.append(f"Overall sentiment for this meeting appears **{sent}** (polarity {pol:.2f}).")
#     if dom:
#         lines.append(f"The dominant emotion detected is **{dom[0]}** ({dom[1]:.2f}).")
#     if history is not None and not history.empty:
#         last = history.tail(3)
#         if "polarity" in last.columns:
#             avg = last["polarity"].mean()
#             diff = pol - avg
#             trend = "higher" if diff > 0 else "lower" if diff < 0 else "similar to"
#             lines.append(f"Compared to the last 3 sessions, polarity is **{trend}** the recent average ({avg:.2f}).")
#     lines.append("Suggested next steps: acknowledge key emotions, reinforce positives, and clarify blockers.")
#     return " ".join(lines)

# def save_pdf(
#     results: Dict,
#     charts: Optional[Dict[str, BytesIO]] = None,
#     pdf_path: Optional[str] = None,
#     return_bytes: bool = False,
#     include_history_charts: Optional[List[BytesIO]] = None
# ):
#     buffer = BytesIO() if return_bytes else None
#     doc = SimpleDocTemplate(buffer if return_bytes else pdf_path, pagesize=A4)
#     styles = getSampleStyleSheet()
#     title = styles["Heading1"]
#     h2 = styles["Heading2"]
#     h3 = styles["Heading3"]
#     body = styles["BodyText"]

#     story = []
#     story.append(Paragraph("Meeting Mood Analyzer — Report", title))
#     story.append(Spacer(1, 12))
#     story.append(Paragraph(datetime.now().strftime("%B %d, %Y %H:%M"), body))
#     story.append(Spacer(1, 24))

#     sent = results.get("sentiment", "N/A")
#     score = results.get("score", 0.0)
#     pol = results.get("polarity", 0.0)
#     sub = results.get("subjectivity", 0.0)
#     story.append(Paragraph(f"<b>Sentiment:</b> {sent}", body))
#     story.append(Paragraph(f"<b>Confidence:</b> {score:.2f}", body))
#     story.append(Paragraph(f"<b>Polarity:</b> {pol:.2f}", body))
#     story.append(Paragraph(f"<b>Subjectivity:</b> {sub:.2f}", body))
#     story.append(PageBreak())

#     story.append(Paragraph("Details", h2))
#     transcript = results.get("transcript")
#     if transcript:
#         story.append(Paragraph("Full Transcript", h3))
#         story.append(Paragraph(transcript.replace("\n", "<br />"), body))
#         story.append(Spacer(1, 12))

#     if results.get("emotions"):
#         story.append(Paragraph("Emotions", h3))
#         for k, v in results["emotions"].items():
#             story.append(Paragraph(f"{k}: {v:.2f}", body))
#         story.append(Spacer(1, 12))

#     if charts:
#         story.append(Paragraph("Visualizations", h2))
#         for caption, buf in charts.items():
#             story.append(Paragraph(caption, h3))
#             story.append(Image(buf, width=15*cm, height=8*cm))
#             story.append(Spacer(1, 18))

#     if include_history_charts:
#         story.append(PageBreak())
#         story.append(Paragraph("Trends Across Sessions", h2))
#         for idx, buf in enumerate(include_history_charts, 1):
#             story.append(Paragraph(f"Trend {idx}", h3))
#             story.append(Image(buf, width=15*cm, height=7*cm))
#             story.append(Spacer(1, 12))

#     story.append(PageBreak())
#     story.append(Paragraph("Conclusion", h2))
#     story.append(Paragraph(_auto_conclusion(results), body))

#     doc.build(story)
#     if return_bytes:
#         buffer.seek(0)
#         return buffer.read()



# Phase 3


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

