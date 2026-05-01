import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from model import BiLSTM, BiLSTM_Attention

print("Loading TEST data...")

# ✅ LOAD TEST DATA ONLY
X = np.load("../data/X_test.npy")
y = np.load("../data/y_test.npy")

X = torch.tensor(X).long()
y = torch.tensor(y).float()

print("Test data:", X.shape)

# Device
device = torch.device("cpu")

# DataLoader
batch_size = 16
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size)

# Load models
print("\nLoading models...")

bilstm = BiLSTM(10000)
bilstm.load_state_dict(torch.load("../models/bilstm_base.pth"))
bilstm.to(device)
bilstm.eval()

bilstm_att = BiLSTM_Attention(10000)
bilstm_att.load_state_dict(torch.load("../models/bilstm_attention.pth"))
bilstm_att.to(device)
bilstm_att.eval()

print("Models loaded")

# Predictions
base_preds = []
att_preds = []

with torch.no_grad():
    for i, (X_batch, _) in enumerate(loader):

        if i % 100 == 0:
            print(f"Batch {i}")

        X_batch = X_batch.to(device)

        out1 = bilstm(X_batch)
        base_preds.extend(out1.cpu().squeeze().numpy())

        out2, _ = bilstm_att(X_batch)
        att_preds.extend(out2.cpu().squeeze().numpy())

# Convert
base_preds = np.array(base_preds)
att_preds = np.array(att_preds)

base_labels = (base_preds > 0.5).astype(int)
att_labels = (att_preds > 0.5).astype(int)

# Metrics
base_acc = accuracy_score(y, base_labels)
att_acc = accuracy_score(y, att_labels)

base_auc = roc_auc_score(y, base_preds)
att_auc = roc_auc_score(y, att_preds)

print("\n📊 FINAL RESULTS (TEST DATA):")
print("BiLSTM Accuracy:", round(base_acc, 4))
print("BiLSTM AUC:", round(base_auc, 4))

print("\nBiLSTM + Attention Accuracy:", round(att_acc, 4))
print("BiLSTM + Attention AUC:", round(att_auc, 4))