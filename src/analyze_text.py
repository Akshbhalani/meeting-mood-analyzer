# src/analyze_text.py

import os
import json
from textblob import TextBlob

def analyze_text(file_path, output_path):
    # Read transcript
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Analyze sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results as JSON
    results = {
        "sentiment": sentiment,
        "polarity_score": polarity
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Text sentiment results saved to: {output_path}")


if __name__ == "__main__":
    input_file = os.path.join("data", "raw", "meeting_transcript.txt")
    output_file = os.path.join("data", "processed", "sentiment_results.json")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Transcript not found: {input_file}. Please create it first.")

    analyze_text(input_file, output_file)
