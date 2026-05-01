import pandas as pd
import re
import nltk
import numpy as np
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize

# =========================
# DOWNLOAD NLTK DATA (FIX)
# =========================
nltk.download("punkt")
nltk.download("punkt_tab")   # important for newer versions

# =========================
# CONFIG
# =========================
MAX_LEN = 200
VOCAB_SIZE = 10000

# =========================
# LOAD DATA
# =========================
fake_df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Documents\fake-news-bilstm-attention\data\Fake.csv")
true_df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Documents\fake-news-bilstm-attention\data\True.csv")

fake_df["label"] = 1
true_df["label"] = 0

df = pd.concat([fake_df, true_df])
df = df[["text", "label"]]

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

print("Dataset loaded:", df.shape)

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# =========================
# TOKENIZATION (SAFE)
# =========================
def safe_tokenize(text):
    try:
        return word_tokenize(text)
    except:
        return text.split()   # fallback

df["tokens"] = df["text"].apply(safe_tokenize)

# =========================
# BUILD VOCAB
# =========================
all_words = []
for tokens in df["tokens"]:
    all_words.extend(tokens)

most_common = Counter(all_words).most_common(VOCAB_SIZE - 1)
word2idx = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}

print("Vocab size:", len(word2idx))

# =========================
# TEXT → SEQUENCE
# =========================
def encode(tokens):
    return [word2idx.get(word, 0) for word in tokens]

df["sequence"] = df["tokens"].apply(encode)

# =========================
# PADDING (200 TOKENS)
# =========================
def pad(seq):
    if len(seq) < MAX_LEN:
        return seq + [0] * (MAX_LEN - len(seq))
    else:
        return seq[:MAX_LEN]

X = np.array([pad(seq) for seq in df["sequence"]])
y = df["label"].values

# =========================
# SAVE
# =========================
np.save("../data/X.npy", X)
np.save("../data/y.npy", y)

with open("../tokenizer.pkl", "wb") as f:
    pickle.dump(word2idx, f)

# =========================
# DONE
# =========================
print("✅ Preprocessing complete")
print("X shape:", X.shape)
print("y shape:", y.shape)