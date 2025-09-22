import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

from analyzer_utils import (
    preprocess_text,
    analyze_textblob,
    analyze_vader,
    evaluate_model,
    generate_wordcloud,
    save_pdf,
)

st.set_page_config(page_title="Meeting Mood Analyzer (Text-Only)", layout="wide")

st.title("📊 Meeting Mood Analyzer (Text-Only Version)")
st.write("Analyze meeting transcripts with TextBlob and VADER baselines.")


# ----------------------------
# Upload transcript file
# ----------------------------
uploaded_file = st.file_uploader("Upload a transcript (.txt)", type="txt")

results = {}
charts = {}

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    st.subheader("📄 Raw Transcript")
    st.text_area("Transcript:", raw_text, height=200)

    # Preprocess
    sentences = [s for s in raw_text.split("\n") if s.strip() != ""]
    clean_sentences = [preprocess_text(s) for s in sentences]

    st.subheader("🔧 Preprocessed Transcript (sample)")
    st.write(clean_sentences[:5])

    # Run analysis sentence by sentence
    tb_results = [analyze_textblob(s) for s in clean_sentences]
    vader_results = [analyze_vader(s) for s in clean_sentences]

    df = pd.DataFrame({
        "sentence": sentences,
        "clean": clean_sentences,
        "tb_label": [r["label"] for r in tb_results],
        "tb_polarity": [r["polarity"] for r in tb_results],
        "vader_label": [r["label"] for r in vader_results],
        "vader_compound": [r["compound"] for r in vader_results],
    })

    # ----------------------------
    # Visualization: Word Cloud
    # ----------------------------
    st.subheader("☁️ Word Cloud")
    wc_buf = generate_wordcloud(" ".join(clean_sentences))
    st.image(wc_buf, caption="Word Cloud", use_container_width=True)
    charts["Word Cloud"] = wc_buf

    # ----------------------------
    # Visualization: Bar Chart
    # ----------------------------
    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x="tb_label", data=df, ax=ax[0], order=["positive", "neutral", "negative"])
    ax[0].set_title("TextBlob Distribution")
    sns.countplot(x="vader_label", data=df, ax=ax[1], order=["positive", "neutral", "negative"])
    ax[1].set_title("VADER Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, caption="Sentiment Distribution", use_container_width=True)
    charts["Distribution"] = buf

    # ----------------------------
    # Visualization: Timeline Plot
    # ----------------------------
    st.subheader("📈 Sentiment Timeline Across Transcript")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["tb_polarity"], label="TextBlob Polarity", marker="o")
    ax.plot(df.index, df["vader_compound"], label="VADER Compound", marker="x")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Trend Over Transcript")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, caption="Timeline of Sentiment", use_container_width=True)
    charts["Timeline"] = buf

    # ----------------------------
    # Visualization: Radar Chart
    # ----------------------------
    st.subheader("📡 Radar Chart (Average Sentiment Comparison)")
    labels = np.array(["Positive", "Neutral", "Negative"])
    tb_counts = df["tb_label"].value_counts(normalize=True).reindex(["positive", "neutral", "negative"]).fillna(0).values
    vader_counts = df["vader_label"].value_counts(normalize=True).reindex(["positive", "neutral", "negative"]).fillna(0).values

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    tb_counts = np.concatenate((tb_counts, [tb_counts[0]]))
    vader_counts = np.concatenate((vader_counts, [vader_counts[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, tb_counts, "o-", linewidth=2, label="TextBlob")
    ax.fill(angles, tb_counts, alpha=0.25)
    ax.plot(angles, vader_counts, "o-", linewidth=2, label="VADER")
    ax.fill(angles, vader_counts, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Radar Chart: Sentiment Distribution Comparison")
    ax.legend(loc="upper right")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, caption="Radar Chart", use_container_width=True)
    charts["Radar"] = buf

    # ----------------------------
    # Evaluation Section
    # ----------------------------
    st.subheader("📈 Evaluation with Labeled Dataset (Optional)")
    st.write("Upload labeled dataset (CSV with columns: text, label).")

    eval_file = st.file_uploader("Upload Labeled Dataset", type="csv")
    if eval_file:
        df_eval = pd.read_csv(eval_file)
        df_eval["clean_text"] = df_eval["text"].apply(preprocess_text)
        df_eval["textblob_pred"] = df_eval["clean_text"].apply(lambda x: analyze_textblob(x)["label"])
        df_eval["vader_pred"] = df_eval["clean_text"].apply(lambda x: analyze_vader(x)["label"])

        st.write("### Evaluation Results")
        tb_metrics, tb_cm_buf = evaluate_model(df_eval["label"], df_eval["textblob_pred"], "TextBlob")
        vader_metrics, vader_cm_buf = evaluate_model(df_eval["label"], df_eval["vader_pred"], "VADER")

        results["metrics"] = [tb_metrics, vader_metrics]
        charts["Confusion Matrix - TextBlob"] = tb_cm_buf
        charts["Confusion Matrix - VADER"] = vader_cm_buf

        st.write(pd.DataFrame(results["metrics"]))

        # ----------------------------
        # PDF Export
        # ----------------------------
        st.sidebar.subheader("📤 Download Report")
        try:
            pdf_bytes = save_pdf(results, charts=charts, return_bytes=True)
            st.sidebar.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="meeting_analysis_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.sidebar.error(f"❌ PDF download failed: {str(e)}")


