import nltk
import spacy
import textblob
from nltk.sentiment import SentimentIntensityAnalyzer

def test_environment():
    print("✅ Testing environment setup for Text-Only Meeting Mood Analyzer...")
    
    try:
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("vader_lexicon")
        print("✅ NLTK resources loaded.")
    except Exception as e:
        print("❌ NLTK error:", e)

    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded.")
    except Exception as e:
        print("❌ spaCy model missing. Run: python -m spacy download en_core_web_sm")

    try:
        from textblob import TextBlob
        blob = TextBlob("Test sentence.")
        print("✅ TextBlob working.")
    except Exception as e:
        print("❌ TextBlob error:", e)

    try:
        sia = SentimentIntensityAnalyzer()
        print("✅ VADER working.")
    except Exception as e:
        print("❌ VADER error:", e)

if __name__ == "__main__":
    test_environment()
