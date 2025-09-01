from transformers import pipeline

# Load pre-trained emotion analysis model
emotion_model = pipeline("text-classification", 
                         model="j-hartmann/emotion-english-distilroberta-base",
                         return_all_scores=True)

# Example sentences (simulate meeting transcript)
texts = [
    "I think this project is going really well!",
    "We are behind schedule and it's stressing me out.",
    "I'm not sure if this is the right direction.",
    "Great job team, we made huge progress today!"
]

# Run predictions
for text in texts:
    results = emotion_model(text)
    print(f"Text: {text}")
    for emotion in results[0]:
        print(f"  {emotion['label']}: {emotion['score']:.4f}")
    print("-" * 50)
