# src/dashboard.py
import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------
# Load meeting summaries
# -----------------------------
def load_meeting_summaries(results_dir="results/meetings"):
    files = [f for f in os.listdir(results_dir) if f.startswith("meeting_summary_") and f.endswith(".json")]
    summaries = []
    for f in files:
        with open(os.path.join(results_dir, f), "r", encoding="utf-8") as infile:
            data = json.load(infile)
            data["file"] = f
            summaries.append(data)
    return summaries

# -----------------------------
# Dashboard UI
# -----------------------------
def main():
    st.set_page_config(page_title="Meeting Mood Analyzer", layout="wide")
    st.title("📊 Meeting Mood Analyzer Dashboard")

    # Load summaries
    results_dir = "results/meetings"
    if not os.path.exists(results_dir):
        st.error(f"No meeting summaries found in {results_dir}. Please run fuse_modalities first.")
        return

    summaries = load_meeting_summaries(results_dir)
    if not summaries:
        st.error("No meeting_summary_*.json files found.")
        return

    df = pd.DataFrame(summaries)

    # Sidebar navigation
    st.sidebar.header("Navigation")
    view = st.sidebar.radio("Go to:", ["Overview", "Individual Meeting", "Trends"])

    # -------------------------
    # Overview Page
    # -------------------------
    if view == "Overview":
        st.header("📌 Overview")
        st.metric("Total Meetings Analyzed", len(df))

        avg_polarity = df["text_polarity"].mean()
        st.metric("Average Text Polarity", f"{avg_polarity:.2f}")

        most_common_emotion = df["video_dominant_emotion"].mode()[0]
        st.metric("Most Common Video Emotion", most_common_emotion)

        st.subheader("Summary Table")
        st.dataframe(df[["file", "text_sentiment", "video_dominant_emotion", "overall_mood"]])

    # -------------------------
    # Individual Meeting View
    # -------------------------
    elif view == "Individual Meeting":
        st.header("📝 Individual Meeting Analysis")
        selected_file = st.selectbox("Select a meeting:", df["file"].tolist())
        meeting = next(m for m in summaries if m["file"] == selected_file)

        st.subheader("Overall Mood")
        st.write(meeting.get("overall_mood", "No summary available."))

        # Text sentiment
        st.subheader("Text Sentiment")
        st.json({
            "sentiment": meeting.get("text_sentiment"),
            "polarity": meeting.get("text_polarity"),
            "subjectivity": meeting.get("text_subjectivity")
        })

        # Video emotions distribution
        st.subheader("Video Emotions Distribution")
        emotions = meeting.get("video_emotion_distribution", {})
        if emotions:
            fig, ax = plt.subplots()
            ax.bar(emotions.keys(), emotions.values())
            ax.set_ylabel("Frequency")
            ax.set_title("Video Emotions")
            st.pyplot(fig)
        else:
            st.info("No video emotion data available.")

        # Word cloud (regenerate from transcript if exists)
        transcript_path = "data/raw/meeting_transcript.txt"
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read()
            wc = WordCloud(width=600, height=400, background_color="white").generate(text)
            st.subheader("Word Cloud from Transcript")
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    # -------------------------
    # Trends Page
    # -------------------------
    elif view == "Trends":
        st.header("📈 Trends Across Meetings")

        # Line chart of polarity
        fig, ax = plt.subplots()
        ax.plot(df["file"], df["text_polarity"], marker="o")
        ax.set_ylabel("Text Polarity")
        ax.set_xlabel("Meeting")
        ax.set_title("Text Polarity Across Meetings")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Bar chart of dominant emotions
        st.subheader("Dominant Video Emotions")
        fig, ax = plt.subplots()
        df["video_dominant_emotion"].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)

# -----------------------------
if __name__ == "__main__":
    main()
