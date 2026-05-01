import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import BiLSTM, BiLSTM_Attention

# =========================
# LOAD DATA
# =========================
print("Loading data...")

X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

# 🔥 OPTIONAL: speed test (uncomment if slow)
# X = X[:5000]
# y = y[:5000]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train).long()
y_train = torch.tensor(y_train).float()

print("Data loaded:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# =========================
# DATALOADER (IMPORTANT)
# =========================
batch_size = 32   # 🔥 smaller = faster + safer

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, name):
    print(f"\n🚀 Training {name} model...")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        model.train()
        total_loss = 0

        print(f"\nStarting Epoch {epoch+1}...")

        for i, (X_batch, y_batch) in enumerate(train_loader):

            # 🔥 DEBUG PRINT (important)
            if i % 100 == 0:
                print(f"Batch {i}")

            # Forward pass
            if name == "attention":
                outputs, _ = model(X_batch)
            else:
                outputs = model(X_batch)

            loss = criterion(outputs.squeeze(), y_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} completed, Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), f"../models/bilstm_{name}.pth")
    print(f"✅ {name} model saved")


# =========================
# TRAIN BOTH MODELS
# =========================
bilstm = BiLSTM(vocab_size=10000)
train_model(bilstm, "base")

bilstm_att = BiLSTM_Attention(vocab_size=10000)
train_model(bilstm_att, "attention")

print("\n🎉 Training complete")