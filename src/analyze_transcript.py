from transformers import pipeline

# Load the model (same as before)
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Read sample transcript
with open("data/text/sample_meetings.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Analyzing meeting transcript...\n")

for line in lines:
    speaker, text = line.split(":", 1)  # split "Name: text"
    results = classifier(text.strip())

    # Get top emotion
    top_emotion = max(results[0], key=lambda x: x['score'])
    print(f"{speaker.strip()} said: {text.strip()}")
    print(f"  → Detected emotion: {top_emotion['label']} ({top_emotion['score']:.2f})")
    print("-" * 50)
