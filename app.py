# import io
# import os
# import json
# from datetime import datetime

# import pandas as pd
# import streamlit as st

# from analyzer_utils import (
#     analyze_text,
#     analyze_audio,
#     analyze_video,
#     plot_wordcloud,
#     plot_word_freq,
#     plot_emotions_bar,
#     plot_emotion_radar,
#     plot_metric_trend,
#     plot_emotion_trend,
#     save_json,
#     save_csv,
#     save_pdf,
#     save_session,
#     load_sessions,
#     SESSIONS_DIR,
#     DEFAULT_EMOTIONS,
# )

# st.set_page_config(page_title="Meeting Mood Analyzer", layout="wide")
# st.title("📊 Meeting Mood Analyzer")

# # ------------------------------
# # Sidebar Navigation
# # ------------------------------
# page = st.sidebar.radio(
#     "Navigate",
#     ["Analyze", "Dashboard", "History & Trends", "Export"],
# )

# # persist last results in session state
# if "last_results" not in st.session_state:
#     st.session_state.last_results = {}
# if "last_charts" not in st.session_state:
#     st.session_state.last_charts = {}

# # ------------------------------
# # Analyze Page
# # ------------------------------
# if page == "Analyze":
#     st.subheader("📝 Analyze a Meeting")

#     option = st.selectbox(
#         "Choose Input Type",
#         ["Text", "Text File Upload", "Audio Upload", "Video Upload"],
#     )

#     results = {}
#     charts = {}

#     # Text
#     if option == "Text":
#         text_input = st.text_area("Enter meeting notes:")
#         if st.button("Analyze Text") and text_input.strip():
#             results = analyze_text(text_input)

#     # Text File
#     elif option == "Text File Upload":
#         uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
#         if uploaded_file and st.button("Analyze File"):
#             text_input = uploaded_file.read().decode("utf-8", errors="ignore")
#             results = analyze_text(text_input)

#     # Audio
#     elif option == "Audio Upload":
#         audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
#         if audio_file and st.button("Analyze Audio"):
#             results = analyze_audio(audio_file)

#     # Video
#     elif option == "Video Upload":
#         video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
#         if video_file and st.button("Analyze Video"):
#             results = analyze_video(video_file)

#     # Show results
#     if results:
#         st.success("Analysis complete.")
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("Sentiment", results.get("sentiment", "N/A"))
#         col2.metric("Confidence", f"{results.get('score', 0):.2f}")
#         col3.metric("Polarity", f"{results.get('polarity', 0):.2f}")
#         col4.metric("Subjectivity", f"{results.get('subjectivity', 0):.2f}")

#         # Visualizations
#         charts = {}

#         # word visuals
#         if results.get("word_freq"):
#             wc_buf = plot_wordcloud(results["word_freq"])
#             bar_buf = plot_word_freq(results["word_freq"])
#             col_w1, col_w2 = st.columns(2)
#             with col_w1:
#                 st.image(wc_buf, caption="Word Cloud", use_container_width=True)
#             with col_w2:
#                 st.image(bar_buf, caption="Word Frequency", use_container_width=True)
#             charts["Word Cloud"] = wc_buf
#             charts["Word Frequency"] = bar_buf

#         # emotion visuals
#         if results.get("emotions"):
#             st.markdown("#### Emotions")
#             emo_filter = st.multiselect(
#                 "Select emotions to display",
#                 options=list(results["emotions"].keys()),
#                 default=list(results["emotions"].keys()),
#             )
#             style = st.radio("Chart style", ["Bar", "Radar"], horizontal=True)
#             if style == "Bar":
#                 emo_buf = plot_emotions_bar(results["emotions"], show=emo_filter)
#                 st.image(emo_buf, caption="Emotions (Bar)", use_container_width=True)
#                 charts["Emotions (Bar)"] = emo_buf
#             else:
#                 radar_buf = plot_emotion_radar(results["emotions"], show=emo_filter)
#                 st.image(radar_buf, caption="Emotions (Radar)", use_container_width=True)
#                 charts["Emotions (Radar)"] = radar_buf

#         # Save current session automatically
#         path = save_session(results)
#         st.caption(f"💾 Session saved: `{os.path.basename(path)}`")

#         # Remember for other pages
#         st.session_state.last_results = results
#         st.session_state.last_charts = charts

# # ------------------------------
# # Dashboard Page (focused view of last analysis)
# # ------------------------------
# elif page == "Dashboard":
#     st.subheader("📈 Dashboard (Latest Analysis)")
#     results = st.session_state.last_results
#     charts = st.session_state.last_charts

#     if not results:
#         st.info("No analysis yet. Go to **Analyze** to run one.")
#     else:
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("Sentiment", results.get("sentiment", "N/A"))
#         col2.metric("Confidence", f"{results.get('score', 0):.2f}")
#         col3.metric("Polarity", f"{results.get('polarity', 0):.2f}")
#         col4.metric("Subjectivity", f"{results.get('subjectivity', 0):.2f}")

#         with st.expander("Transcript", expanded=False):
#             st.write(results.get("transcript", ""))

#         # Show one chart per concept
#         c1, c2 = st.columns(2)
#         if "Word Cloud" in charts:
#             with c1:
#                 st.image(charts["Word Cloud"], caption="Word Cloud", use_container_width=True)
#         if "Word Frequency" in charts:
#             with c2:
#                 st.image(charts["Word Frequency"], caption="Word Frequency", use_container_width=True)

#         # emotion: let user choose style (one visualization at a time)
#         if results.get("emotions"):
#             st.markdown("#### Emotions")
#             emo_filter = st.multiselect(
#                 "Select emotions",
#                 options=list(results["emotions"].keys()),
#                 default=list(results["emotions"].keys()),
#                 key="dash_emo_filter",
#             )
#             style = st.radio("Chart style", ["Bar", "Radar"], horizontal=True, key="dash_emo_style")
#             if style == "Bar":
#                 emo_buf = plot_emotions_bar(results["emotions"], show=emo_filter)
#                 st.image(emo_buf, caption="Emotions (Bar)", use_container_width=True)
#             else:
#                 radar_buf = plot_emotion_radar(results["emotions"], show=emo_filter)
#                 st.image(radar_buf, caption="Emotions (Radar)", use_container_width=True)

# # ------------------------------
# # History & Trends
# # ------------------------------
# elif page == "History & Trends":
#     st.subheader("📉 Trends Across Sessions")
#     df = load_sessions(SESSIONS_DIR)

#     if df.empty:
#         st.info("No saved sessions yet.")
#     else:
#         # filters
#         colf1, colf2 = st.columns(2)
#         with colf1:
#             start = st.date_input("Start date", value=df["timestamp"].min().date() if "timestamp" in df else datetime.now().date())
#         with colf2:
#             end = st.date_input("End date", value=df["timestamp"].max().date() if "timestamp" in df else datetime.now().date())
#         if "timestamp" in df:
#             m = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
#             df = df[m]

#         if df.empty:
#             st.warning("No sessions in selected date range.")
#         else:
#             st.dataframe(df[["timestamp", "sentiment", "polarity", "subjectivity", "score"]], use_container_width=True)

#             # metric trends
#             st.markdown("#### Metric Trends")
#             mt_col1, mt_col2 = st.columns(2)
#             pol_buf = plot_metric_trend(df, "polarity")
#             sub_buf = plot_metric_trend(df, "subjectivity")
#             with mt_col1:
#                 st.image(pol_buf, caption="Polarity Trend", use_container_width=True)
#             with mt_col2:
#                 st.image(sub_buf, caption="Subjectivity Trend", use_container_width=True)

#             # score trend
#             score_buf = plot_metric_trend(df, "score")
#             st.image(score_buf, caption="Confidence Score Trend", use_container_width=True)

#             # emotion trend (single selection)
#             st.markdown("#### Emotion Trend")
#             emo_cols = [c for c in df.columns if c.startswith("emo_")]
#             available = [c.replace("emo_", "") for c in emo_cols]
#             pick = st.selectbox("Select an emotion", options=available or DEFAULT_EMOTIONS)
#             if pick:
#                 et_buf = plot_emotion_trend(df, pick)
#                 st.image(et_buf, caption=f"Trend: {pick}", use_container_width=True)

#             # keep some history charts for PDF export
#             hist_charts = [pol_buf, sub_buf, score_buf]
#             if pick:
#                 hist_charts.append(et_buf)
#             st.session_state.history_charts = hist_charts

# # ------------------------------
# # Export page
# # ------------------------------
# elif page == "Export":
#     st.subheader("📤 Export Results")
#     results = st.session_state.last_results
#     charts = st.session_state.last_charts

#     if not results:
#         st.info("No latest analysis to export. Run something in **Analyze**.")
#     else:
#         st.write("Choose your export format:")

#         # JSON
#         try:
#             json_bytes = json.dumps(results, indent=4).encode("utf-8")
#             st.download_button(
#                 label="📥 Download JSON",
#                 data=json_bytes,
#                 file_name="analysis_results.json",
#                 mime="application/json",
#             )
#         except Exception as e:
#             st.error(f"❌ JSON export failed: {e}")

#         # CSV
#         try:
#             df = pd.DataFrame([results])
#             csv_buffer = io.StringIO()
#             df.to_csv(csv_buffer, index=False)
#             st.download_button(
#                 label="📥 Download CSV",
#                 data=csv_buffer.getvalue(),
#                 file_name="analysis_results.csv",
#                 mime="text/csv",
#             )
#         except Exception as e:
#             st.error(f"❌ CSV export failed: {e}")

#         # PDF
#         include_trends = st.checkbox("Include history trend charts (if available)", value=True)
#         hist_charts = st.session_state.get("history_charts", None) if include_trends else None
#         try:
#             pdf_bytes = save_pdf(results, charts=charts, return_bytes=True, include_history_charts=hist_charts)
#             st.download_button(
#                 label="📥 Download PDF Report",
#                 data=pdf_bytes,
#                 file_name="meeting_analysis_report.pdf",
#                 mime="application/pdf",
#             )
#         except Exception as e:
#             st.error(f"❌ PDF export failed: {e}")

# ---------------------------------------------------------------------------------

# import streamlit as st
# from analyzer_utils import (
#     analyze_text,
#     analyze_audio,
#     analyze_video,
#     plot_wordcloud,
#     plot_word_freq,
#     plot_emotions_bar,
#     plot_emotion_radar,
#     save_json,
#     save_csv,
#     save_pdf
# )
# import io
# import json
# import pandas as pd

# st.set_page_config(page_title="Meeting Mood Analyzer", layout="wide")
# st.title("📊 Meeting Mood Analyzer Dashboard")

# option = st.sidebar.selectbox(
#     "Choose Input Type",
#     ["Text", "Text File Upload", "Audio Upload", "Video Upload"]
# )

# results = {}
# charts = {}

# # ------------------------------
# # Text Input
# # ------------------------------
# if option == "Text":
#     st.subheader("📝 Text Sentiment Analysis")
#     text_input = st.text_area("Enter meeting notes:")
#     if st.button("Analyze Text") and text_input.strip():
#         results = analyze_text(text_input)

# # ------------------------------
# # Text File Upload
# # ------------------------------
# elif option == "Text File Upload":
#     st.subheader("📄 Upload a Text File")
#     uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
#     if uploaded_file and st.button("Analyze File"):
#         text_input = uploaded_file.read().decode("utf-8")
#         results = analyze_text(text_input)

# # ------------------------------
# # Audio Upload
# # ------------------------------
# elif option == "Audio Upload":
#     st.subheader("🎤 Audio Sentiment Analysis")
#     audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
#     if audio_file and st.button("Analyze Audio"):
#         results = analyze_audio(audio_file)

# # ------------------------------
# # Video Upload
# # ------------------------------
# elif option == "Video Upload":
#     st.subheader("📹 Video Emotion Analysis")
#     video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
#     if video_file and st.button("Analyze Video"):
#         results = analyze_video(video_file)

# # ------------------------------
# # Dashboard & Visualization
# # ------------------------------
# if results:
#     st.subheader("📈 Dashboard Results")

#     if "error" in results:
#         st.error(results["error"])
#     else:
#         charts = {}

#         # Sentiment (for text/audio/video)
#         if "sentiment" in results:
#             st.metric("Predicted Sentiment", results["sentiment"])
#             st.metric("Confidence", round(results.get("score", 0.0), 3))
#             st.metric("Polarity", round(results.get("polarity", 0.0), 3))
#             st.metric("Subjectivity", round(results.get("subjectivity", 0.0), 3))

#             if results.get("word_freq"):
#                 wc_buf = plot_wordcloud(results["word_freq"])
#                 bar_buf = plot_word_freq(results["word_freq"])
#                 st.image(wc_buf, caption="Word Cloud", use_container_width=True)
#                 st.image(bar_buf, caption="Word Frequency", use_container_width=True)
#                 charts.update({"Word Cloud": wc_buf, "Word Frequency": bar_buf})

#         # Emotions (for video, simulated)
#         if "emotions" in results:
#             emo_buf = plot_emotions_bar(results["emotions"])
#             radar_buf = plot_emotion_radar(results["emotions"])
#             st.image(emo_buf, caption="Detected Emotions", use_container_width=True)
#             st.image(radar_buf, caption="Emotion Radar", use_container_width=True)
#             charts.update({"Emotions": emo_buf, "Emotion Radar": radar_buf})

#     # --------------------------
#     # Direct Download Section
#     # --------------------------
#     st.sidebar.subheader("📤 Download Results")

#     # JSON Download
#     try:
#         json_bytes = json.dumps(results, indent=4).encode("utf-8")
#         st.sidebar.download_button(
#             label="📥 Download JSON",
#             data=json_bytes,
#             file_name="analysis_results.json",
#             mime="application/json"
#         )
#     except Exception as e:
#         st.sidebar.error(f"❌ JSON download failed: {str(e)}")

#     # CSV Download
#     try:
#         df = pd.DataFrame([results])
#         csv_buffer = io.StringIO()
#         df.to_csv(csv_buffer, index=False)
#         st.sidebar.download_button(
#             label="📥 Download CSV",
#             data=csv_buffer.getvalue(),
#             file_name="analysis_results.csv",
#             mime="text/csv"
#         )
#     except Exception as e:
#         st.sidebar.error(f"❌ CSV download failed: {str(e)}")

#     # PDF Download (with charts)
#     try:
#         pdf_bytes = save_pdf(results, charts=charts, return_bytes=True)
#         st.sidebar.download_button(
#             label="📥 Download PDF Report",
#             data=pdf_bytes,
#             file_name="meeting_analysis_report.pdf",
#             mime="application/pdf"
#         )
#     except Exception as e:
#         st.sidebar.error(f"❌ PDF download failed: {str(e)}")


# -------------------------------------------------------------------


import streamlit as st
from analyzer_utils import (
    analyze_text,
    analyze_audio,
    analyze_video,
    plot_wordcloud,
    plot_word_freq,
    plot_emotions_bar,
    plot_emotion_radar,
    save_json,
    save_csv,
    save_pdf
)
import io
import json
import pandas as pd

st.set_page_config(page_title="Meeting Mood Analyzer", layout="wide")
st.title("📊 Meeting Mood Analyzer Dashboard")

option = st.sidebar.selectbox(
    "Choose Input Type",
    ["Text", "Text File Upload", "Audio Upload", "Video Upload"]
)

results = {}
charts = {}

# ------------------------------
# Text Input
# ------------------------------
if option == "Text":
    st.subheader("📝 Text Sentiment Analysis")
    text_input = st.text_area("Enter meeting notes:")
    if st.button("Analyze Text") and text_input.strip():
        results = analyze_text(text_input)

# ------------------------------
# Text File Upload
# ------------------------------
elif option == "Text File Upload":
    st.subheader("📄 Upload a Text File")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file and st.button("Analyze File"):
        text_input = uploaded_file.read().decode("utf-8")
        results = analyze_text(text_input)

# ------------------------------
# Audio Upload
# ------------------------------
elif option == "Audio Upload":
    st.subheader("🎤 Audio Sentiment Analysis")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if audio_file and st.button("Analyze Audio"):
        results = analyze_audio(audio_file)

# ------------------------------
# Video Upload
# ------------------------------
elif option == "Video Upload":
    st.subheader("📹 Video Emotion Analysis")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file and st.button("Analyze Video"):
        results = analyze_video(video_file)

# ------------------------------
# Dashboard & Visualization
# ------------------------------
if results:
    st.subheader("📈 Dashboard Results")

    if "error" in results:
        st.error(results["error"])
    else:
        charts = {}

        # Sentiment (for text/audio/video)
        if "sentiment" in results:
            st.metric("Predicted Sentiment", results["sentiment"])
            st.metric("Confidence", round(results.get("score", 0.0), 3))
            st.metric("Polarity", round(results.get("polarity", 0.0), 3))
            st.metric("Subjectivity", round(results.get("subjectivity", 0.0), 3))

            if results.get("word_freq"):
                wc_buf = plot_wordcloud(results["word_freq"])
                bar_buf = plot_word_freq(results["word_freq"])
                st.image(wc_buf, caption="Word Cloud", use_container_width=True)
                st.image(bar_buf, caption="Word Frequency", use_container_width=True)
                charts.update({"Word Cloud": wc_buf, "Word Frequency": bar_buf})

        # Emotions (for video, simulated)
        if "emotions" in results:
            emo_buf = plot_emotions_bar(results["emotions"])
            radar_buf = plot_emotion_radar(results["emotions"])
            st.image(emo_buf, caption="Detected Emotions", use_container_width=True)
            st.image(radar_buf, caption="Emotion Radar", use_container_width=True)
            charts.update({"Emotions": emo_buf, "Emotion Radar": radar_buf})

    # --------------------------
    # Direct Download Section
    # --------------------------
    st.sidebar.subheader("📤 Download Results")

    # JSON Download
    try:
        json_bytes = json.dumps(results, indent=4).encode("utf-8")
        st.sidebar.download_button(
            label="📥 Download JSON",
            data=json_bytes,
            file_name="analysis_results.json",
            mime="application/json"
        )
    except Exception as e:
        st.sidebar.error(f"❌ JSON download failed: {str(e)}")

    # CSV Download
    try:
        df = pd.DataFrame([results])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="analysis_results.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.sidebar.error(f"❌ CSV download failed: {str(e)}")

    # PDF Download (with charts)
    try:
        pdf_bytes = save_pdf(results, charts=charts, return_bytes=True)
        st.sidebar.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name="meeting_analysis_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.sidebar.error(f"❌ PDF download failed: {str(e)}")


