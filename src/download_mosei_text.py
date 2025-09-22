# src/download_mosei_text.py
from datasets import load_dataset
import pandas as pd
import os

def map_label(score):
    # CMU-MOSEI sentiment is -3 to +3
    if score <= -1:
        return "negative"
    elif score >= 1:
        return "positive"
    else:
        return "neutral"

def main(out_csv="data/test/mosei_sample.csv", sample_size=1000):
    print("📥 Downloading CMU-MOSEI text subset...")
    dataset = load_dataset("multimodal-datasets/CMU-MOSEI", "text")

    # Use only 'train' split for sampling
    df = pd.DataFrame(dataset["train"])
    df = df[["text", "label"]].rename(columns={"label": "score"})

    # Map scores (-3 to +3) into labels
    df["label"] = df["score"].apply(map_label)

    # Drop missing or empty text
    df = df[df["text"].str.strip() != ""].dropna(subset=["text"])

    # Sample smaller subset for local testing
    df_sample = df.sample(n=sample_size, random_state=42)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_sample[["text", "label"]].to_csv(out_csv, index=False)

    print(f"✅ Saved {len(df_sample)} samples to {out_csv}")

if __name__ == "__main__":
    main()
