
# **Meeting Mood Analyzer**

The **Meeting Mood Analyzer** is a Python-based application that detects and analyzes the emotional tone of meetings using **multimodal data sources** (audio, video, and text). It helps organizations visualize emotional trends, identify communication bottlenecks, and generate actionable insights for better collaboration.  

---

## **Overview**

Meetings play a critical role in organizational productivity. However, emotional dynamics during meetings often remain unnoticed. This project aims to:  

- Automatically **detect emotions** from audio, video, and text.  
- Provide **visual insights** like emotion timelines, word clouds, and distribution charts.  
- Generate **reports** (HTML/JSON) with summaries and recommendations.  

---

## **Features**

- 🎤 **Audio Emotion Detection**: Uses voice features (MFCC, pitch, energy).  
- 📹 **Video Analysis**: Extracts frames, detects faces, classifies expressions.  
- 📝 **Text Sentiment Analysis**: NLP-based analysis of transcripts.  
- 📊 **Visualization**: Word clouds, emotion distribution, and timelines.  
- 📑 **Reporting**: HTML and JSON summaries with insights.  

---

## **Installation**

### **Prerequisites**
- Python **3.8+**  
- Git  

<!-- ### **Steps** -->

# Clone the repository
git clone https://github.com/Akshbhalani/meeting-mood-analyzer.git
cd meeting-mood-analyzer

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

---

## **Usage**

Run the analyzer with:
    python app.py \

  --audio data/audio/meeting.wav \

  --video data/video/meeting.mp4 \

  --text data/text/transcript.txt \
  
  --output results/



CLI Options

--audio : Path to audio file (.wav)

--video : Path to video file (.mp4)

--text : Path to transcript (.txt)

--output: Folder to save results


---

## **Configuration**

- config.json: Stores project settings (e.g., model selection, thresholds, output paths).

- Environment Variables: For API keys (speech-to-text, cloud storage).

- Command-line Flags: Override config at runtime.

---

## **Visualization & Reports**

- Charts: Emotion distributions (pie/bar/histogram).

- Word Clouds: Highlight frequent emotional words.

- Timeline Graphs: Emotion trends across the meeting.

- Reports: HTML & JSON summaries with recommendations.


