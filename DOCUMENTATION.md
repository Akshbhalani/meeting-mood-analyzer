# Meeting Mood Analyzer Documentation

## 1. Project Overview

The Meeting Mood Analyzer is a Python-based application designed to analyze the emotional tone of meetings using multimodal data sources such as audio, video, and text. The system processes meeting recordings and transcripts to detect emotions, visualize trends, and generate comprehensive reports. This documentation provides a detailed explanation of the system, its components, and the workflow.

## 2. Motivation & Objectives

Meetings are a crucial part of organizational communication. Understanding the emotional dynamics during meetings can help improve team collaboration, identify issues, and enhance productivity. The objectives of this project are:

- To automatically detect emotions from meeting data.
- To visualize emotional trends and distributions.
- To generate actionable reports for stakeholders.

## 3. System Architecture & Process Flow Diagram

The system consists of several modules:

- **Data Collection**: Gathers audio, video, and text data from meetings.
- **Preprocessing**: Cleans and prepares data for analysis.
- **Emotion Detection**: Applies machine learning and NLP techniques to detect emotions.
- **Visualization**: Generates charts and word clouds to represent emotional trends.
- **Reporting**: Compiles results into HTML/JSON reports.

### Process Flow Diagram

Below is a description of the process flow. You can use this to create a diagram in draw.io or mermaid:

```
[Start] --> [Data Collection] --> [Preprocessing] --> [Emotion Detection] --> [Visualization] --> [Reporting] --> [End]

Data Collection:
    |-- Audio (recorded_audio.wav)
    |-- Video Frames (captured_frames/)
    |-- Text (transcripts)

Preprocessing:
    |-- Noise reduction (audio)
    |-- Frame extraction (video)
    |-- Text cleaning (transcripts)

Emotion Detection:
    |-- Audio emotion analysis
    |-- Video facial emotion recognition
    |-- Text sentiment analysis

Visualization:
    |-- Emotion distribution charts
    |-- Word clouds
    |-- Timeline graphs

Reporting:
    |-- HTML/JSON reports
    |-- Summary and insights
```

## 4. Data Collection


The system collects data from multiple sources, each requiring specific handling and storage:

- **Audio**: Meeting audio is recorded and stored as WAV files. These files are typically captured using microphones during online or in-person meetings. The audio data is crucial for detecting vocal emotions such as tone, pitch, and intensity.

- **Video**: Frames are captured from meeting videos and stored as images in the `captured_frames/` directory. Video data allows for facial emotion recognition, body language analysis, and context understanding. Frame extraction is performed at regular intervals to ensure representative sampling.

- **Text**: Transcripts are generated from audio using speech-to-text algorithms or provided directly by meeting participants. Text data is used for sentiment analysis and keyword extraction.

Data is organized in the `data/` folder, with subfolders for each type, ensuring easy access and modular processing. The structure supports scalability for large meetings and multiple sessions.

## 5. Data Preprocessing

Preprocessing is a critical step to ensure data quality and reliability for emotion analysis. Each data type undergoes specific preprocessing steps:

### Audio Preprocessing
- **Noise Reduction**: Removes background noise using digital filters (e.g., spectral gating).
- **Segmentation**: Splits audio into manageable chunks for analysis.
- **Feature Extraction**: Extracts features such as MFCCs (Mel Frequency Cepstral Coefficients), pitch, energy, and zero-crossing rate.

### Video Preprocessing
- **Frame Extraction**: Selects frames at regular intervals or based on scene changes.
- **Face Detection**: Uses algorithms (e.g., Haar cascades, Dlib, or deep learning models) to locate faces in frames.
- **Normalization**: Adjusts brightness, contrast, and scales images for consistent analysis.

### Text Preprocessing
- **Cleaning**: Removes filler words, punctuation, and corrects spelling errors.
- **Tokenization**: Splits text into words or sentences.
- **Stopword Removal**: Eliminates common words that do not contribute to sentiment.
- **Lemmatization/Stemming**: Reduces words to their base forms.

## 6. Emotion Detection Algorithms

The core of the system is emotion detection, which leverages machine learning and natural language processing techniques:

### Audio Emotion Analysis
- **Feature-Based Classification**: Uses extracted audio features to classify emotions (e.g., happy, sad, angry) using models like SVM, Random Forest, or deep neural networks.
- **Pre-trained Models**: May use models trained on emotion-labeled datasets (e.g., RAVDESS, EmoDB).

### Video Facial Emotion Recognition
- **Face Landmark Detection**: Identifies key facial points (eyes, mouth, eyebrows).
- **Expression Classification**: Applies CNNs (Convolutional Neural Networks) to classify facial expressions into emotions.
- **Temporal Analysis**: Tracks emotion changes over time using sequential models (e.g., LSTM).

### Text Sentiment Analysis
- **Lexicon-Based Methods**: Uses sentiment dictionaries to score text.
- **Machine Learning Models**: Trains classifiers (e.g., Logistic Regression, Naive Bayes) on labeled sentiment data.
- **Deep Learning**: Utilizes transformers (e.g., BERT) for advanced sentiment and emotion classification.

## 7. Visualization & Reporting

Visualization is essential for interpreting results and communicating insights:

- **Emotion Distribution Charts**: Pie charts, bar graphs, and histograms show the proportion of each emotion detected.
- **Word Clouds**: Visualize frequently used words in meeting transcripts, highlighting emotional keywords.
- **Timeline Graphs**: Plot emotion confidence and changes over time, revealing trends and critical moments.

Reports are generated in HTML and JSON formats, including:
- **Summary Statistics**: Overall emotional tone, dominant emotions, and participant engagement.
- **Session Analysis**: Breakdowns by meeting segment or participant.
- **Recommendations**: Actionable insights for improving future meetings.

## 8. User Interface & Usage

The application is designed for ease of use:

- **Command-Line Interface (CLI)**: Users can run the main script (`app.py`) with arguments specifying input data and output preferences.
- **Configuration**: Settings for data paths, model selection, and visualization options are managed via configuration files or command-line flags.
- **Output**: Results are saved in the `results/` folder, with visualizations and reports accessible for review.

### Example Usage
```bash
python app.py --audio data/audio/meeting.wav --video data/video/meeting.mp4 --text data/text/transcript.txt --output results/
```

## 9. Testing & Validation

Testing ensures the reliability and accuracy

- **Unit Tests**: Core functions are tested using scripts like `test_env.py`.
- **Validation Datasets**: The system is evaluated on labeled datasets to measure emotion detection accuracy.
- **Performance Metrics**: Precision, recall, F1-score, and confusion matrices are used to assess model performance.
- **Error Analysis**: Misclassifications are analyzed to improve algorithms and preprocessing.

## 10. Future Work & Improvements

Potential enhancements include:

- **Real-Time Analysis**: Implementing live emotion detection during meetings.
- **Multilingual Support**: Expanding text analysis to multiple languages.
- **Advanced Visualization**: Interactive dashboards and deeper analytics.
- **Integration**: Connecting with meeting platforms (e.g., Zoom, Teams) for automated data collection.
- **User Feedback**: Incorporating feedback mechanisms for continuous improvement.

## 11. Detailed Process Flow Diagram

Below is a more detailed description for creating a process diagram:

```
graph TD
    A[Start] --> B[Data Collection]
    B --> C[Preprocessing]
    C --> D[Emotion Detection]
    D --> E[Visualization]
    E --> F[Reporting]
    F --> G[End]

    B --> B1[Audio]
    B --> B2[Video]
    B --> B3[Text]
    C --> C1[Noise Reduction]
    C --> C2[Frame Extraction]
    C --> C3[Text Cleaning]
    D --> D1[Audio Analysis]
    D --> D2[Facial Recognition]
    D --> D3[Sentiment Analysis]
    E --> E1[Charts]
    E --> E2[Word Clouds]
    E --> E3[Timelines]
    F --> F1[HTML Report]
    F --> F2[JSON Summary]
```

This diagram can be rendered using mermaid or similar tools for visual representation.

---


This documentation provides a comprehensive guide to the Meeting Mood Analyzer project, covering every step from data collection to reporting and future improvements. For further details, refer to the source code and experiment with the provided scripts and datasets.

---

## 12. Installation Guide

Follow these steps to set up the project environment:

1. Clone the repository:
    ```bash
    git clone <repo-url>
    ```
2. Install Python (recommended version: 3.8+).
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. (Optional) Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
5. Run the application as described in the usage section.

## 13. API Reference

Document all major functions, classes, and modules. Example:

- `analyzer_utils.py`: Contains utility functions for data processing and emotion analysis.
- `app.py`: Main application logic, orchestrates workflow.
- `test_env.py`: Testing scripts for validation.

For each function, include:
- Name, parameters, return type, description, example usage.

## 14. Configuration Details

Explain configuration files, environment variables, and customization options:

- `config.json` or YAML files for model settings, data paths, and output preferences.
- Environment variables for API keys or platform integration.
- Command-line flags for runtime options.

## 15. Sample Data & Results

Include example input files and expected outputs:

- Sample audio, video, and transcript files in the `data/` folder.
- Example results in the `results/` folder (charts, reports, word clouds).
- Screenshots or snippets of HTML/JSON reports.

## 16. Troubleshooting

Common issues and solutions:

- **Dependency errors**: Ensure all packages in `requirements.txt` are installed.
- **File not found**: Check data paths and file names.
- **Model errors**: Verify model files and configuration settings.
- **Permission issues**: Run with appropriate user rights.

## 17. Glossary

Define technical terms and acronyms used in the project:

- **MFCC**: Mel Frequency Cepstral Coefficients
- **CNN**: Convolutional Neural Network
- **LSTM**: Long Short-Term Memory
- **Sentiment Analysis**: Process of determining emotional tone in text
- **Speaker Diarization**: Identifying who spoke when in audio

## 18. References

List research papers, datasets, and libraries used:

- RAVDESS, EmoDB (emotion datasets)
- OpenCV, Dlib (computer vision libraries)
- NLTK, spaCy, BERT (NLP libraries)
- Relevant academic papers on emotion recognition

## 19. Contribution Guide

Instructions for contributing to the project:

- Fork the repository and create a feature branch.
- Follow code style and documentation standards.
- Submit pull requests with clear descriptions.
- Report issues and suggest enhancements via GitHub Issues.

## 20. Changelog

Track major updates and improvements:

- v1.0: Initial release with core features
- v1.1: Added advanced visualization and reporting
- v1.2: Integrated speaker diarization and multilingual support
  
---  
