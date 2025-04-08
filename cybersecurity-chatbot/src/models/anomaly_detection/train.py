import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from src.models.anomaly_detection.models.autoencoder import Autoencoder
from src.models.anomaly_detection.processors.traffic_processor import preprocess_traffic_data


# Load paths
TRAIN_PATH = "D:/Chatbot datasets/cybersecurity-chatbot/datasets/intrusion_detection/nslkdd/raw/KDDTrain+.txt"
TEST_PATH = "D:/Chatbot datasets/cybersecurity-chatbot/datasets/intrusion_detection/nslkdd/raw/KDDTest+.txt"

print("🔄 Loading dataset...")
train_df = pd.read_csv(TRAIN_PATH, header=None)
test_df = pd.read_csv(TEST_PATH, header=None)

# Only train on normal traffic (label 0)
train_df = train_df[train_df[41] == "normal"]

# Preprocess
print("⚙️ Preprocessing...")
train_data, encoders, scaler = preprocess_traffic_data(train_df, fit_scaler=True)
test_data, _, _ = preprocess_traffic_data(test_df, fit_scaler=False, scaler=scaler)

# Extract labels for evaluation
test_labels = test_data["label"].values
test_data = test_data.drop(columns=["label"])
train_data = train_data.drop(columns=["label"])

# Convert to tensors
train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

# Train model
input_dim = train_tensor.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("🏋️ Training autoencoder...")
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    outputs = model(train_tensor)
    loss = criterion(outputs, train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

# Evaluation
print("🧪 Evaluating model...")
model.eval()
with torch.no_grad():
    reconstructed = model(test_tensor)
    mse = torch.mean((reconstructed - test_tensor) ** 2, dim=1).numpy()

# Find best threshold for F1
thresholds = np.linspace(min(mse), max(mse), 100)
best_f1 = 0
best_thresh = 0
for t in thresholds:
    preds = (mse > t).astype(int)
    f1 = f1_score(test_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"🔍 Optimal Threshold: {best_thresh:.6f}")
print(f"✅ Best F1 Score: {best_f1:.4f}")
print(f"📈 Final AUC: {roc_auc_score(test_labels, mse):.4f}")

# Save model
torch.save(model.state_dict(), "best_anomaly_model.pth")
print("✅ Model saved successfully!")
