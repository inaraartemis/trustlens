import torch
import torch.nn as nn
import os
from transformers import BertModel

print("Skipping training... saving model")

# =========================
# DEFINE SAME MODEL (IMPORTANT)
# =========================
class HybridModel(nn.Module):
    def __init__(self, feature_dim=5001):  # 5000 TF-IDF + 1 POS
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768 + feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.pooler_output
        combined = torch.cat((cls, features), dim=1)
        return self.sigmoid(self.fc(combined))


# =========================
# CREATE MODEL INSTANCE
# =========================
model = HybridModel()

# =========================
# CREATE MODELS FOLDER
# =========================
os.makedirs("../models", exist_ok=True)

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "../models/hybrid_model.pth")

print("✅ Model saved successfully!")