import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from features import build_features
import torch.nn as nn
import os

df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\Documents\bert-hybrid-fake-news\data\combined.csv")

texts = df["text"].tolist()[:2000]
labels = df["label"].values[:2000]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=200, return_tensors="pt")

test_enc = encode(X_test)

test_features = build_features(X_test)
test_features = torch.tensor(test_features).float()

class HybridModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768 + feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.pooler_output
        combined = torch.cat((cls, features), dim=1)
        return self.sigmoid(self.fc(combined))

model = HybridModel(test_features.shape[1])
model.load_state_dict(torch.load("../models/hybrid_model.pth"))
model.eval()

with torch.no_grad():
    outputs = model(test_enc["input_ids"], test_enc["attention_mask"], test_features)

preds = outputs.squeeze().numpy()
pred_labels = (preds > 0.5).astype(int)

os.makedirs("../outputs", exist_ok=True)

errors = []
for i in range(len(y_test)):
    if y_test[i] != pred_labels[i]:
        errors.append((X_test[i], y_test[i], pred_labels[i]))

with open("../outputs/errors.txt", "w", encoding="utf-8") as f:
    for text, true, pred in errors[:20]:
        f.write(f"TEXT: {text}\nTRUE: {true} | PRED: {pred}\n\n")

print("✅ Errors saved to outputs/errors.txt")