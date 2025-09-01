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

# # --------------------------
# # Paths & constants
# # --------------------------
# SESSIONS_DIR = os.path.join("results", "sessions")
# os.makedirs(SESSIONS_DIR, exist_ok=True)

# DEFAULT_EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]

# # --------------------------
# # Lightweight text analysis (no Transformers)
# # --------------------------
# def _simple_word_freq(text: str, top_n: int = 30) -> Dict[str, int]:
#     """
#     Very lightweight tokenizer + frequency counter (no external downloads).
#     """
#     stop = set("""
#         the a an and or is are was were be been being i me my we our you your he she it
#         they them this that to of in for on with as at by from up down out over under about
#         not no yes do did done have has had can could would should may might will just
#         if then there here when where who what why how
#     """.split())
#     tokens = []
#     # simple alnum split
#     for raw in text.lower().replace("\n", " ").split():
#         tok = "".join(ch for ch in raw if ch.isalnum())
#         if tok and tok not in stop and len(tok) > 2:
#             tokens.append(tok)
#     freq = {}
#     for t in tokens:
#         freq[t] = freq.get(t, 0) + 1
#     # top-n
#     return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n])


# def _simple_sentiment(text: str) -> Tuple[str, float, float, float]:
#     """
#     Super-simple lexicon-based sentiment approximation so we don't pull heavy models.
#     Returns (label, score, polarity, subjectivity).
#     polarity ~ [-1, 1], score ~ [0,1] confidence-like
#     """
#     positive_words = set("""
#         good great excellent amazing awesome happy joy love nice progress success productive
#         positive outstanding fantastic brilliant wonderful helpful satisfied improved win
#     """.split())
#     negative_words = set("""
#         bad poor terrible awful sad angry upset fail problem delay blocked issue negative
#         worse worst disappointing unproductive unhappy stuck conflict
#     """.split())

#     words = [w.strip(".,!?").lower() for w in text.split()]
#     pos = sum(1 for w in words if w in positive_words)
#     neg = sum(1 for w in words if w in negative_words)

#     total = max(len(words), 1)
#     polarity = (pos - neg) / max(pos + neg, 1)  # scale by sentiment words only
#     subjectivity = min((pos + neg) / total * 4.0, 1.0)  # heuristic
#     score = min(abs(polarity) + 0.2, 1.0)
#     label = "Neutral"
#     if polarity > 0.1:
#         label = "Positive"
#     elif polarity < -0.1:
#         label = "Negative"

#     return label, float(score), float(polarity), float(subjectivity)


# def analyze_text(text: str) -> Dict:
#     """
#     Text sentiment + simple word freq + a tiny mock emotion estimate (optional).
#     """
#     label, score, polarity, subjectivity = _simple_sentiment(text or "")
#     wf = _simple_word_freq(text or "")
#     # a tiny mock emotion distribution based on sentiment direction
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
#     """
#     Simulated audio analysis: pretend we've transcribed the audio.
#     """
#     # If you later add real STT, place it here and feed result to analyze_text().
#     fake_transcript = "This is a simulated transcript extracted from the uploaded audio."
#     return analyze_text(fake_transcript)


# def analyze_video(video_file) -> Dict:
#     """
#     Simulated video analysis producing an emotions distribution and a mock transcript.
#     """
#     emotions = {"happy": 0.45, "neutral": 0.35, "sad": 0.1, "surprise": 0.07, "angry": 0.03}
#     return {
#         "emotions": emotions,
#         "transcript": "Simulated transcript derived from the uploaded video.",
#         # add thin sentiment computed from transcript for consistency:
#         **{k: v for k, v in analyze_text("Simulated transcript derived from the uploaded video.").items()
#            if k in ("sentiment", "score", "polarity", "subjectivity")}
#     }

# # --------------------------
# # Persistence (sessions) & history
# # --------------------------
# def save_session(results: Dict, directory: str = SESSIONS_DIR) -> str:
#     os.makedirs(directory, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     uid = uuid.uuid4().hex[:8]
#     path = os.path.join(directory, f"session_{ts}_{uid}.json")
#     # enrich with metadata
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
#         full = os.path.join(directory, name)
#         try:
#             with open(full, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 rows.append(data)
#         except Exception:
#             continue
#     if not rows:
#         return pd.DataFrame()
#     df = pd.DataFrame(rows)
#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#         df = df.sort_values("timestamp")
#     # normalize emotions (dict) into columns if present
#     if "emotions" in df.columns:
#         emo_df = df["emotions"].apply(lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series())
#         emo_df = emo_df.fillna(0.0)
#         for col in emo_df.columns:
#             df[f"emo_{col}"] = emo_df[col]
#     return df

# # --------------------------
# # Plot helpers (return BytesIO buffers for Streamlit & PDF)
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
#     # empty placeholder
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
#         # compare with last 3 sessions average
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
#     """
#     Builds a multi-page PDF with cover, details, charts, and conclusion.
#     """
#     buffer = BytesIO() if return_bytes else None
#     doc = SimpleDocTemplate(buffer if return_bytes else pdf_path, pagesize=A4)
#     styles = getSampleStyleSheet()
#     title = styles["Heading1"]
#     h2 = styles["Heading2"]
#     h3 = styles["Heading3"]
#     body = styles["BodyText"]

#     story = []

#     # --- Cover Page ---
#     story.append(Paragraph("Meeting Mood Analyzer — Report", title))
#     story.append(Spacer(1, 12))
#     story.append(Paragraph(datetime.now().strftime("%B %d, %Y %H:%M"), body))
#     story.append(Spacer(1, 24))
#     # quick summary bullets
#     sent = results.get("sentiment", "N/A")
#     score = results.get("score", 0.0)
#     pol = results.get("polarity", 0.0)
#     sub = results.get("subjectivity", 0.0)
#     story.append(Paragraph(f"<b>Sentiment:</b> {sent}", body))
#     story.append(Paragraph(f"<b>Confidence:</b> {score:.2f}", body))
#     story.append(Paragraph(f"<b>Polarity:</b> {pol:.2f}", body))
#     story.append(Paragraph(f"<b>Subjectivity:</b> {sub:.2f}", body))
#     story.append(PageBreak())

#     # --- Details (Transcript + Emotions) ---
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

#     # --- Charts (current) ---
#     if charts:
#         story.append(Paragraph("Visualizations", h2))
#         for caption, buf in charts.items():
#             story.append(Paragraph(caption, h3))
#             story.append(Image(buf, width=15*cm, height=8*cm))
#             story.append(Spacer(1, 18))

#     # --- History Charts (optional) ---
#     if include_history_charts:
#         story.append(PageBreak())
#         story.append(Paragraph("Trends Across Sessions", h2))
#         for idx, buf in enumerate(include_history_charts, 1):
#             story.append(Paragraph(f"Trend {idx}", h3))
#             story.append(Image(buf, width=15*cm, height=7*cm))
#             story.append(Spacer(1, 12))

#     # --- Conclusion ---
#     story.append(PageBreak())
#     story.append(Paragraph("Conclusion", h2))
#     # we can’t load history here (no df parameter), the app will pass text if needed
#     story.append(Paragraph(_auto_conclusion(results), body))

#     doc.build(story)
#     if return_bytes:
#         buffer.seek(0)
#         return buffer.read()

# ---------------------------------------------------------------



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

# # load HuggingFace pipeline lazily (lightweight)
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
#     label = "Neutral"
#     score = 0.5
#     if hf_sentiment:
#         try:
#             res = hf_sentiment(text[:512])[0]  # truncate long text
#             label = res["label"]
#             score = float(res["score"])
#         except Exception:
#             pass

#     # Harmonize label
#     if "POS" in label.upper():
#         label = "Positive"
#     elif "NEG" in label.upper():
#         label = "Negative"
#     else:
#         label = "Neutral"

#     return label, score, polarity, subjectivity


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
#         text = "This is a simulated transcript extracted from the uploaded audio."

#     return analyze_text(text)


# def analyze_video(video_file) -> Dict:
#     # Extract audio
#     transcript = None
#     try:
#         clip = VideoFileClip(video_file)
#         audio_path = video_file + "_temp.wav"
#         clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

#         transcript = analyze_audio(audio_path)["transcript"]
#         os.remove(audio_path)
#     except Exception:
#         transcript = "Simulated transcript derived from the uploaded video."

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



# ------------------------------------------------------------------------------



import os
import json
import uuid
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# Sentiment + NLP
from textblob import TextBlob
from transformers import pipeline

# Audio/Video
import speech_recognition as sr
from moviepy.editor import VideoFileClip

# --------------------------
# Paths & constants
# --------------------------
SESSIONS_DIR = os.path.join("results", "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

DEFAULT_EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]

# Load HuggingFace pipeline lazily
try:
    hf_sentiment = pipeline("sentiment-analysis")
except Exception:
    hf_sentiment = None

# --------------------------
# Hybrid Sentiment Analysis
# --------------------------
def _hybrid_sentiment(text: str) -> Tuple[str, float, float, float]:
    """
    Combine TextBlob + HuggingFace for sentiment.
    Returns (label, score, polarity, subjectivity).
    """
    if not text.strip():
        return "Neutral", 0.0, 0.0, 0.0

    # TextBlob
    tb = TextBlob(text)
    polarity = float(tb.sentiment.polarity)
    subjectivity = float(tb.sentiment.subjectivity)

    # HuggingFace (fallback if not available)
    label = None
    score = 0.0
    if hf_sentiment:
        try:
            res = hf_sentiment(text[:512])[0]  # truncate long text
            label = res.get("label", None)
            score = float(res.get("score", 0.0))
        except Exception:
            label = None

    # Harmonize label
    if label:
        label_lower = label.lower()
        if label_lower in ["pos", "positive", "label_1"]:
            label = "Positive"
        elif label_lower in ["neg", "negative", "label_0"]:
            label = "Negative"
        else:
            label = None  # fallback to TextBlob
    if not label:
        # Fallback: use polarity
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

    return label, score, polarity, subjectivity

# --------------------------
# Word Frequency
# --------------------------
def _simple_word_freq(text: str, top_n: int = 30) -> Dict[str, int]:
    stop = set("""
        the a an and or is are was were be been being i me my we our you your he she it
        they them this that to of in for on with as at by from up down out over under about
        not no yes do did done have has had can could would should may might will just
        if then there here when where who what why how
    """.split())
    tokens = []
    for raw in text.lower().replace("\n", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if tok and tok not in stop and len(tok) > 2:
            tokens.append(tok)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

# --------------------------
# Core analyzers
# --------------------------
def analyze_text(text: str) -> Dict:
    label, score, polarity, subjectivity = _hybrid_sentiment(text or "")
    wf = _simple_word_freq(text or "")

    # Tiny heuristic emotion map
    if polarity >= 0.1:
        emotions = {"happy": 0.6, "surprise": 0.15, "neutral": 0.25}
    elif polarity <= -0.1:
        emotions = {"sad": 0.5, "angry": 0.2, "neutral": 0.3}
    else:
        emotions = {"neutral": 0.7, "happy": 0.15, "sad": 0.15}

    return {
        "sentiment": label,
        "score": score,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "word_freq": wf,
        "emotions": emotions,
        "transcript": text,
    }


def analyze_audio(audio_file) -> Dict:
    recognizer = sr.Recognizer()
    text = None
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
    except Exception:
        text = "Simulated transcript: " \
               "audio could not be processed, using TextBlob fallback sentiment."

    return analyze_text(text)


def analyze_video(video_file) -> Dict:
    transcript = None
    try:
        clip = VideoFileClip(video_file)
        audio_path = video_file + "_temp.wav"
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        transcript = analyze_audio(audio_path)["transcript"]
        os.remove(audio_path)
    except Exception:
        transcript = "Simulated transcript: video could not be processed, using TextBlob fallback sentiment."

    return analyze_text(transcript)

# --------------------------
# Persistence & history
# --------------------------
def save_session(results: Dict, directory: str = SESSIONS_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    path = os.path.join(directory, f"session_{ts}_{uid}.json")
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **results
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_sessions(directory: str = SESSIONS_DIR) -> pd.DataFrame:
    if not os.path.isdir(directory):
        return pd.DataFrame()
    rows = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    if "emotions" in df.columns:
        emo_df = df["emotions"].apply(lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series())
        emo_df = emo_df.fillna(0.0)
        for col in emo_df.columns:
            df[f"emo_{col}"] = emo_df[col]
    return df

# --------------------------
# Plot helpers
# --------------------------
def _save_current_fig_to_buf(dpi: int = 150) -> BytesIO:
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="PNG", dpi=dpi, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

def plot_wordcloud(word_freq: Dict[str, int]) -> BytesIO:
    wc = WordCloud(width=900, height=450, background_color="white")
    if word_freq:
        wc = wc.generate_from_frequencies(word_freq)
        img = wc.to_image()
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    plt.figure(figsize=(6, 3))
    plt.text(0.5, 0.5, "No words", ha="center", va="center")
    return _save_current_fig_to_buf()

def plot_word_freq(word_freq: Dict[str, int], top_n: int = 20) -> BytesIO:
    items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Word Frequency")
    plt.ylabel("Count")
    return _save_current_fig_to_buf()

def plot_emotions_bar(emotions: Dict[str, float], show: Optional[List[str]] = None) -> BytesIO:
    emo = emotions or {}
    if show:
        emo = {k: v for k, v in emo.items() if k in show}
    labels = list(emo.keys())
    values = list(emo.values())
    plt.figure(figsize=(6, 3.6))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.title("Emotions (Bar)")
    return _save_current_fig_to_buf()

def plot_emotion_radar(emotions: Dict[str, float], show: Optional[List[str]] = None) -> BytesIO:
    emo = emotions or {}
    if show:
        emo = {k: v for k, v in emo.items() if k in show}
    if not emo:
        plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, "No emotions", ha="center", va="center")
        return _save_current_fig_to_buf()

    labels = list(emo.keys())
    values = list(emo.values())
    angles = [n / float(len(labels)) * 2.0 * 3.14159265 for n in range(len(labels))]
    values = values + values[:1]
    angles = angles + angles[:1]

    plt.figure(figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Emotions (Radar)")
    return _save_current_fig_to_buf()

def plot_metric_trend(df: pd.DataFrame, metric: str) -> BytesIO:
    if df.empty or metric not in df.columns:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
        return _save_current_fig_to_buf()
    plt.figure(figsize=(7, 3.8))
    plt.plot(df["timestamp"], df[metric], marker="o")
    plt.title(f"Trend: {metric}")
    plt.xlabel("Time")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    return _save_current_fig_to_buf()

def plot_emotion_trend(df: pd.DataFrame, emotion: str) -> BytesIO:
    col = f"emo_{emotion}"
    return plot_metric_trend(df, col)

# --------------------------
# Export helpers
# --------------------------
def save_json(results: Dict, json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_csv(results: Dict, csv_path: str) -> None:
    df = pd.DataFrame([results])
    df.to_csv(csv_path, index=False)

def _dominant_emotion(emotions: Dict[str, float]) -> Optional[Tuple[str, float]]:
    if not emotions:
        return None
    k = max(emotions, key=lambda x: emotions[x])
    return k, emotions[k]

def _auto_conclusion(results: Dict, history: Optional[pd.DataFrame] = None) -> str:
    lines = []
    sent = results.get("sentiment", "Unknown")
    pol = results.get("polarity", 0.0)
    dom = _dominant_emotion(results.get("emotions", {}))
    lines.append(f"Overall sentiment for this meeting appears **{sent}** (polarity {pol:.2f}).")
    if dom:
        lines.append(f"The dominant emotion detected is **{dom[0]}** ({dom[1]:.2f}).")
    if history is not None and not history.empty:
        last = history.tail(3)
        if "polarity" in last.columns:
            avg = last["polarity"].mean()
            diff = pol - avg
            trend = "higher" if diff > 0 else "lower" if diff < 0 else "similar to"
            lines.append(f"Compared to the last 3 sessions, polarity is **{trend}** the recent average ({avg:.2f}).")
    lines.append("Suggested next steps: acknowledge key emotions, reinforce positives, and clarify blockers.")
    return " ".join(lines)

def save_pdf(
    results: Dict,
    charts: Optional[Dict[str, BytesIO]] = None,
    pdf_path: Optional[str] = None,
    return_bytes: bool = False,
    include_history_charts: Optional[List[BytesIO]] = None
):
    buffer = BytesIO() if return_bytes else None
    doc = SimpleDocTemplate(buffer if return_bytes else pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    title = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Meeting Mood Analyzer — Report", title))
    story.append(Spacer(1, 12))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y %H:%M"), body))
    story.append(Spacer(1, 24))

    sent = results.get("sentiment", "N/A")
    score = results.get("score", 0.0)
    pol = results.get("polarity", 0.0)
    sub = results.get("subjectivity", 0.0)
    story.append(Paragraph(f"<b>Sentiment:</b> {sent}", body))
    story.append(Paragraph(f"<b>Confidence:</b> {score:.2f}", body))
    story.append(Paragraph(f"<b>Polarity:</b> {pol:.2f}", body))
    story.append(Paragraph(f"<b>Subjectivity:</b> {sub:.2f}", body))
    story.append(PageBreak())

    story.append(Paragraph("Details", h2))
    transcript = results.get("transcript")
    if transcript:
        story.append(Paragraph("Full Transcript", h3))
        story.append(Paragraph(transcript.replace("\n", "<br />"), body))
        story.append(Spacer(1, 12))

    if results.get("emotions"):
        story.append(Paragraph("Emotions", h3))
        for k, v in results["emotions"].items():
            story.append(Paragraph(f"{k}: {v:.2f}", body))
        story.append(Spacer(1, 12))

    if charts:
        story.append(Paragraph("Visualizations", h2))
        for caption, buf in charts.items():
            story.append(Paragraph(caption, h3))
            story.append(Image(buf, width=15*cm, height=8*cm))
            story.append(Spacer(1, 18))

    if include_history_charts:
        story.append(PageBreak())
        story.append(Paragraph("Trends Across Sessions", h2))
        for idx, buf in enumerate(include_history_charts, 1):
            story.append(Paragraph(f"Trend {idx}", h3))
            story.append(Image(buf, width=15*cm, height=7*cm))
            story.append(Spacer(1, 12))

    story.append(PageBreak())
    story.append(Paragraph("Conclusion", h2))
    story.append(Paragraph(_auto_conclusion(results), body))

    doc.build(story)
    if return_bytes:
        buffer.seek(0)
        return buffer.read()
