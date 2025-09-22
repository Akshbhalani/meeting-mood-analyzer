# src/download_sst2_sample.py
from datasets import load_dataset
import pandas as pd
import os

def main(out_csv="data/test/sst2_sample.csv", sample_size=1000):
    print("📥 Downloading SST-2 (Stanford Sentiment Treebank v2)...")
    dataset = load_dataset("glue", "sst2")

    # Convert to dataframe
    df = pd.DataFrame(dataset["train"])
    df = df.rename(columns={"sentence": "text", "label": "score"})
    df["label"] = df["score"].map({0: "negative", 1: "positive"})

    # Drop empty text
    df = df[df["text"].str.strip() != ""].dropna(subset=["text"])

    # Sample smaller subset for local testing
    df_sample = df.sample(n=sample_size, random_state=42)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_sample[["text", "label"]].to_csv(out_csv, index=False)

    print(f"✅ Saved {len(df_sample)} samples to {out_csv}")

if __name__ == "__main__":
    main()
