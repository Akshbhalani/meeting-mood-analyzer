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
