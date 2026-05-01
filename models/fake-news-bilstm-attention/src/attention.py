import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from model import BiLSTM_Attention

# =========================
# LOAD TOKENIZER (DICT)
# =========================
with open("../tokenizer.pkl", "rb") as f:
    word2idx = pickle.load(f)

# =========================
# SAMPLE TEXT
# =========================
text = "Breaking news government announces new policy affecting economy"

tokens = word_tokenize(text.lower())

# Convert to sequence
sequence = [word2idx.get(word, 0) for word in tokens]

# Padding
MAX_LEN = 200
if len(sequence) < MAX_LEN:
    sequence += [0] * (MAX_LEN - len(sequence))
else:
    sequence = sequence[:MAX_LEN]

X = torch.tensor([sequence]).long()

# =========================
# LOAD MODEL
# =========================
model = BiLSTM_Attention(10000)
model.load_state_dict(torch.load("../models/bilstm_attention.pth"))
model.eval()

# =========================
# GET ATTENTION
# =========================
with torch.no_grad():
    output, attention_weights = model(X)

attention_weights = attention_weights.squeeze().numpy()

# Take only real tokens length
attention_weights = attention_weights[:len(tokens)]

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 3))
plt.bar(tokens, attention_weights)
plt.xticks(rotation=45)
plt.title("Attention Weights")
plt.tight_layout()

plt.savefig("../outputs/attention_plot.png")
plt.show()

print("✅ Attention plot saved")