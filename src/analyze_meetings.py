import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_meeting_summaries(meetings_dir):
    """Load all JSON summaries from meetings_dir into a DataFrame."""
    records = []
    for file in Path(meetings_dir).glob("meeting_summary_*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Try extracting safe values
        record = {
            "file": file.name,
            "date": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "text_sentiment": data.get("text_sentiment"),
            "video_dominant_emotion": data.get("video_dominant_emotion"),
            "overall_mood": data.get("overall_mood"),
            "text_polarity": data.get("text_polarity", None),
            "text_subjectivity": data.get("text_subjectivity", None),
        }
        records.append(record)

    if not records:
        raise FileNotFoundError(f"No meeting_summary_*.json files found in {meetings_dir}")

    return pd.DataFrame(records)

def plot_trends(df, results_dir):
    """Generate plots for multi-session trends."""
    os.makedirs(results_dir, exist_ok=True)

    # Line plot of text polarity trend
    plt.figure(figsize=(8, 4))
    plt.plot(df["date"], df["text_polarity"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Text Sentiment Polarity Over Meetings")
    plt.xlabel("Meeting Date")
    plt.ylabel("Polarity (-1 negative → +1 positive)")
    plt.tight_layout()
    polarity_plot = os.path.join(results_dir, "trend_text_polarity.png")
    plt.savefig(polarity_plot)
    plt.close()

    # Bar plot of video dominant emotions
    plt.figure(figsize=(6, 4))
    df["video_dominant_emotion"].value_counts().plot(kind="bar")
    plt.title("Video Dominant Emotions Across Meetings")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    emotion_plot = os.path.join(results_dir, "video_emotion_distribution.png")
    plt.savefig(emotion_plot)
    plt.close()

    return polarity_plot, emotion_plot

def generate_html_report(df, polarity_plot, emotion_plot, output_file):
    """Generate HTML report combining text + plots."""
    html = f"""
    <html>
    <head>
        <title>Meeting Trends Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            th {{ background-color: #f4f4f4; }}
            img {{ max-width: 100%; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Meeting Trends Report</h1>
        <h2>Meeting Records</h2>
        {df.to_html(index=False)}

        <h2>Text Polarity Trend</h2>
        <img src="{os.path.basename(polarity_plot)}">

        <h2>Video Emotion Distribution</h2>
        <img src="{os.path.basename(emotion_plot)}">
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    meetings_dir = "results/meetings"
    results_dir = "results/trends"
    os.makedirs(results_dir, exist_ok=True)

    df = load_meeting_summaries(meetings_dir)

    polarity_plot, emotion_plot = plot_trends(df, results_dir)

    output_html = os.path.join(results_dir, "meetings_summary.html")
    generate_html_report(df, polarity_plot, emotion_plot, output_html)

    print(f"[INFO] Multi-session trends report saved to: {output_html}")


if __name__ == "__main__":
    main()
