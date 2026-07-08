import csv, string, sys
from pathlib import Path

def normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation)).strip()
    text = " ".join(text.split())
    return text

def build_training_vocab(rows):
    training_voc = set()
    for row in rows:
        if row["split"] == "train":
            phrase = row["text"]
            norm_phrase = normalize(phrase)
            words = set(norm_phrase.split())
            training_voc = training_voc | words
    
    return training_voc

def catch_leaks(rows, train_vocab):
    leaks = []
    for row in rows:
        if row["split"] == "test" and row["eval_set"].strip() == "A":
            phrase = row["text"]
            norm_phrase = normalize(phrase)
            words = set(norm_phrase.split())
            for word in words:
                if word in train_vocab:
                    leaks.append((phrase, word))
        
    return leaks





if __name__ == "__main__":
    PROMPTS = Path(__file__).parent.parent / "dataset_design" / "word_prompts.txt"
    rows = list(csv.DictReader(open(PROMPTS)))
    training_vocab = build_training_vocab(rows)
    leaks = catch_leaks(rows, training_vocab)
    print(leaks)
    if not leaks:
        print("No Set A leakage.")
    sys.exit(1 if leaks else 0)
