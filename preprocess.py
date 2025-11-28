# src/classify.py
import os
import pandas as pd
from transformers import pipeline

MODEL = "s-nlp/roberta_toxicity_classifier"
TOXIC_WORDS_FILE = "toxic_words.txt"


def load_dictionary(file_path=TOXIC_WORDS_FILE):
    """
    Load toxic words/phrases from a text file, one per line.
    Returns a set of lowercase strings.
    """
    if not os.path.exists(file_path):
        print(f"[WARN] Dictionary file {file_path} not found. Continuing without it.")
        return set()

    with open(file_path, "r", encoding="utf-8") as f:
        words = [w.strip().lower() for w in f if w.strip()]

    toxic_set = set(words)
    print(f"Loaded {len(toxic_set)} toxic dictionary entries from {file_path}")
    return toxic_set


def classify(input_csv="data/toxic_lines.csv", output_csv="data/toxic_results.csv"):
    # read preprocessed comments
    df = pd.read_csv(input_csv)

    # load transformer classifier
    classifier = pipeline(
        "text-classification",
        model=MODEL,
        return_all_scores=True
    )

    # load dictionary of explicit toxic words/phrases
    toxic_dict = load_dictionary()

    all_results = []

    for text in df["text"]:
        text_str = str(text)
        text_lower = text_str.lower()

        # run transformer model
        scores = classifier(text_str)[0]   # list of {"label": "...", "score": ...}

        toxic_score = [s["score"] for s in scores if s["label"] == "toxic"][0]
        neutral_score = [s["score"] for s in scores if s["label"] == "neutral"][0]

        # rule-based dictionary flag
        dict_flag = any(word in text_lower for word in toxic_dict)

        # hybrid decision:
        # if model says toxic OR dictionary finds a strong toxic word → Toxic
        if toxic_score > neutral_score or dict_flag:
            prediction = "Toxic"
        else:
            prediction = "Non-toxic"

        all_results.append({
            "text": text_str,
            "toxic_score": toxic_score,
            "neutral_score": neutral_score,
            "prediction": prediction,
            "dict_flag": dict_flag  # optional extra info
        })

    out_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("Classification saved to:", output_csv)


if __name__ == "__main__":
    classify()
