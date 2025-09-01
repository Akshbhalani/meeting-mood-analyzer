# import os
# import json
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import webbrowser
# from datetime import datetime
# from textblob import TextBlob
# from collections import Counter


# def load_text_sentiment(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         sentiment = json.load(f)

#     # Handle both old and new versions
#     if isinstance(sentiment, dict) and "sentiment" in sentiment:
#         return {
#             "sentiment": sentiment["sentiment"],
#             "polarity": sentiment.get("polarity", 0),
#         }
#     elif isinstance(sentiment, dict):
#         return sentiment
#     else:
#         raise ValueError("Unexpected format in sentiment_results.json")


# def load_video_emotions(file_path):
#     df = pd.read_csv(file_path)
#     return df


# def find_latest_file(folder, pattern):
#     files = glob.glob(os.path.join(folder, pattern))
#     if not files:
#         raise FileNotFoundError(f"No files found in {folder} matching {pattern}")
#     return max(files, key=os.path.getctime)


# def plot_emotion_trend(df, results_dir):
#     plt.figure(figsize=(10, 5))
#     df["Dominant Emotion"].value_counts().plot(kind="bar", color="skyblue")
#     plt.title("Overall Video Emotion Distribution")
#     plt.xlabel("Emotion")
#     plt.ylabel("Count")
#     plot_path = os.path.join(results_dir, "video_emotion_distribution.png")
#     plt.savefig(plot_path)
#     plt.close()
#     return plot_path


# def plot_emotion_over_time(df, results_dir):
#     plt.figure(figsize=(12, 6))
#     plt.plot(df["Frame"], df["Confidence"], label="Confidence", color="green", alpha=0.7)
#     plt.title("Emotion Confidence Over Time (Frames)")
#     plt.xlabel("Frame")
#     plt.ylabel("Confidence")
#     plt.legend()
#     plot_path = os.path.join(results_dir, "emotion_confidence_over_time.png")
#     plt.savefig(plot_path)
#     plt.close()
#     return plot_path


# def generate_wordcloud(text_file, results_dir):
#     with open(text_file, "r", encoding="utf-8") as f:
#         text = f.read()
#     wc = WordCloud(width=800, height=400, background_color="white").generate(text)
#     plot_path = os.path.join(results_dir, "wordcloud.png")
#     wc.to_file(plot_path)
#     return plot_path


# def generate_html_report(output_json, plots, html_report):
#     with open(output_json, "r", encoding="utf-8") as f:
#         summary = json.load(f)

#     html_content = f"""
#     <html>
#     <head>
#         <title>Meeting Mood Report</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; margin: 20px; }}
#             h1 {{ color: #2c3e50; }}
#             img {{ max-width: 600px; margin: 10px 0; }}
#             .section {{ margin-bottom: 30px; }}
#         </style>
#     </head>
#     <body>
#         <h1>Meeting Mood Report</h1>
#         <div class="section">
#             <h2>Summary</h2>
#             <pre>{json.dumps(summary, indent=4)}</pre>
#         </div>
#         <div class="section">
#             <h2>Video Emotion Distribution</h2>
#             <img src="{plots['emotion_dist']}" alt="Video Emotion Distribution">
#         </div>
#         <div class="section">
#             <h2>Emotion Confidence Over Time</h2>
#             <img src="{plots['emotion_time']}" alt="Emotion Confidence Over Time">
#         </div>
#         <div class="section">
#             <h2>Word Cloud (Transcript)</h2>
#             <img src="{plots['wordcloud']}" alt="Word Cloud">
#         </div>
#     </body>
#     </html>
#     """

#     with open(html_report, "w", encoding="utf-8") as f:
#         f.write(html_content)

#     print(f"Interactive report saved to: {html_report}")
#     webbrowser.open(f"file://{os.path.abspath(html_report)}")


# # --- Text Sentiment Analyzer ---
# def analyze_text_sentiment(transcript: str) -> dict:
#     """Analyze transcript text using TextBlob sentiment analysis."""
#     if not transcript.strip():
#         return {"polarity": 0, "subjectivity": 0}
    
#     blob = TextBlob(transcript)
#     return {
#         "polarity": blob.sentiment.polarity,
#         "subjectivity": blob.sentiment.subjectivity
#     }

# # --- Video emotion file finder ---
# def find_video_emotion_file(video_folder: str) -> str:
#     """Find the most recent video emotion CSV file in the given folder."""
#     if not os.path.exists(video_folder):
#         raise FileNotFoundError(f"Video folder not found: {video_folder}")
    
#     csv_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".csv")]
#     if not csv_files:
#         raise FileNotFoundError(f"No video emotion CSV files found in {video_folder}")
    
#     # Pick the latest file
#     latest_file = max(csv_files, key=os.path.getctime)
#     return latest_file

# # --- Video emotions loader ---
# def load_video_emotions(video_file: str) -> pd.DataFrame:
#     """Load emotions CSV file as DataFrame."""
#     return pd.read_csv(video_file)




# def summarize_video_emotions(video_df):
#     """
#     Summarize emotions from video CSV (handles different column naming styles).
#     """
#     possible_cols = ["emotion", "predicted_emotion", "dominant_emotion", "Dominant Emotion", "Emotion"]
#     found_col = None

#     # Normalize column names for matching
#     for col in video_df.columns:
#         if col.strip().lower() in [c.lower() for c in possible_cols]:
#             found_col = col
#             break

#     if not found_col:
#         raise ValueError(
#             f"[ERROR] Could not find an emotion column. Expected one of {possible_cols}, "
#             f"but found {video_df.columns.tolist()}"
#         )

#     print(f"[INFO] Using column '{found_col}' for video emotion summarization")

#     # Count frequencies
#     emotion_counts = video_df[found_col].value_counts().to_dict()

#     # Normalize (percentage)
#     total = sum(emotion_counts.values())
#     emotion_percentages = {k: round(v / total * 100, 2) for k, v in emotion_counts.items()}

#     return {
#         "counts": emotion_counts,
#         "percentages": emotion_percentages,
#         "dominant": max(emotion_counts, key=emotion_counts.get)
#     }





# def fuse_modalities(text_file, video_folder, output_json, results_dir):
#     # --- Load transcript ---
#     transcript = ""
#     if os.path.exists(text_file):
#         with open(text_file, "r", encoding="utf-8") as f:
#             transcript = f.read()
#     else:
#         print(f"[WARNING] Transcript file not found: {text_file}")

#     # --- Analyze text sentiment ---
#     text_sentiment = analyze_text_sentiment(transcript) if transcript else {"polarity": 0, "subjectivity": 0}

#     # --- Process video emotions ---
#     video_file = find_video_emotion_file(video_folder)
#     print(f"[INFO] Using video file: {video_file}")
#     video_emotions = load_video_emotions(video_file)

#     # --- Create combined summary ---
#     summary = {
#         "text_polarity": text_sentiment.get("polarity", 0),
#         "text_subjectivity": text_sentiment.get("subjectivity", 0),
#         "transcript": transcript,   # ✅ Now transcript is defined
#         "video_emotions": summarize_video_emotions(video_emotions),
#     }

#     # --- Save summary ---
#     os.makedirs(results_dir, exist_ok=True)
#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=4)

#     print(f"Meeting mood summary saved to: {output_json}")

#     # Generate plots
#     plots = {
#         "emotion_dist": plot_emotion_trend(video_emotions, results_dir),
#         "emotion_time": plot_emotion_over_time(video_emotions, results_dir),
#         "wordcloud": generate_wordcloud("data/raw/meeting_transcript.txt", results_dir),
#     }

#     # Generate HTML report
#     html_report = os.path.join(results_dir, "meeting_report.html")
#     generate_html_report(output_json, plots, html_report)


#      # ✅ Create subfolder for meeting history
#     meetings_dir = os.path.join(results_dir, "meetings")
#     os.makedirs(meetings_dir, exist_ok=True)

#     # ✅ Generate timestamped filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     meeting_summary_file = os.path.join(meetings_dir, f"meeting_summary_{timestamp}.json")

#     # ✅ Save main summary as timestamped file
#     with open(meeting_summary_file, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=4)

#     print(f"[INFO] Meeting mood summary saved to: {meeting_summary_file}")
#     print(f"[INFO] Plots saved in: {results_dir}")


# if __name__ == "__main__":
#     text_file = os.path.join("data", "processed", "sentiment_results.json")
#     video_folder = os.path.join("..", "data", "video")
#     output_json = os.path.join("results", "meeting_summary.json")
#     results_dir = "results"

#     fuse_modalities(text_file, video_folder, output_json, results_dir)




# -------------------------------------------------------------------------------------------------------
import os
import json
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

TEXT_FILE = os.path.join(DATA_DIR, "text", "transcript.txt")
AUDIO_FILE = os.path.join(DATA_DIR, "audio", "audio_analysis.json")
VIDEO_FILE = os.path.join(DATA_DIR, "video", "video_analysis.json")

FUSED_JSON = os.path.join(RESULTS_DIR, "fused_results.json")
REPORT_HTML = os.path.join(RESULTS_DIR, "report.html")

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_json(path):
    """Safely load JSON file if it exists."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")
            return {}
    else:
        print(f"[WARN] File not found: {path}")
        return {}


def process_text():
    """Dummy text sentiment analysis if transcript exists."""
    if not os.path.exists(TEXT_FILE):
        return {
            "dominant": "N/A",
            "counts": {},
            "percentages": {}
        }

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    if not transcript:
        return {
            "dominant": "N/A",
            "counts": {},
            "percentages": {}
        }

    # For now, just simulate (later you may add real NLP model here)
    return {
        "dominant": "neutral",
        "counts": {"neutral": 1},
        "percentages": {"neutral": 100}
    }


def generate_report(summary):
    """Generate a simple HTML report from summary."""
    html_content = f"""
    <html>
    <head><title>Meeting Mood Analyzer Report</title></head>
    <body>
        <h1>Meeting Mood Analyzer Report</h1>
        <p><b>Generated on:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <h2>Summary</h2>
        <pre>{json.dumps(summary, indent=4)}</pre>
    </body>
    </html>
    """
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[INFO] Report saved: {REPORT_HTML}")


def main():
    # Load all modality results
    text_sentiment = process_text()
    audio_emotions = load_json(AUDIO_FILE)
    video_emotions = load_json(VIDEO_FILE)

    # Fuse into one JSON
    fused = {
        "text_sentiment": text_sentiment,
        "audio_emotions": audio_emotions or {"dominant": "N/A", "counts": {}, "percentages": {}},
        "video_emotions": video_emotions or {"dominant": "N/A", "counts": {}, "percentages": {}},
        "timestamp": datetime.now().isoformat()
    }

    with open(FUSED_JSON, "w", encoding="utf-8") as f:
        json.dump(fused, f, indent=4)

    print(f"[INFO] JSON summary saved: {FUSED_JSON}")

    generate_report(fused)


if __name__ == "__main__":
    main()
