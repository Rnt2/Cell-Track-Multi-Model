# test_model_on_external_dataset.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from Config import *

# ==== BiLSTM with Attention ====
class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        weights = self.dropout(weights)
        context = torch.sum(weights.unsqueeze(-1) * lstm_out, dim=1)
        return context, weights

class BiLSTMAttnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim*2, dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attn(lstm_out)
        return self.fc(context), attn_weights

# ==== Load External Test Dataset ====
TEST_PATH = f"{GENERATED_DIR}/cart_test_trajectory_{SEQ_LEN}.npz"
data = np.load(TEST_PATH,allow_pickle=True)
X_test, y_test_raw, track_ids = data["X"], data["y"], data["track_ids"]

# === Transpose shape if needed ===
if X_test.shape[1] == FEATURE_LEN and X_test.shape[2] == SEQ_LEN:
    X_test = np.transpose(X_test, (0, 2, 1))

X_test = torch.tensor(X_test, dtype=torch.float32)

# === Label Encode using predefined order ===
le = LabelEncoder()
le.fit([0, 1, 2])  # Ensure consistent encoding
y_test = le.transform(y_test_raw)
y_test = torch.tensor(y_test, dtype=torch.long)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

# ==== Load Model ====
model = BiLSTMAttnModel(input_dim=FEATURE_LEN, hidden_dim=128, output_dim=3, dropout=0.5)
model.load_state_dict(torch.load(f"{MODEL_DIR}/model_best_{SEQ_LEN}.pth"))
model.eval()

# ==== Evaluate ====
y_true, y_pred, probs, attns = [], [], [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        out, att = model(Xb)
        y_true.extend(yb.tolist())
        y_pred.extend(out.argmax(dim=1).tolist())
        probs.extend(F.softmax(out, dim=1).tolist())
        attns.extend(att.numpy())

# ==== Report ====
print("📊 External Test Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

# ==== Confusion Matrix ====
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (External Test Set)")
plt.savefig(f"{SEQ_RESULT_DIR}/ext_confusion_matrix.png")
plt.close()

# ==== ROC Curve ====
y_bin = label_binarize(y_true, classes=[0, 1, 2])
probs = np.array(probs)
for i in range(3):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve (External Test Set)")
plt.legend()
plt.savefig(f"{SEQ_RESULT_DIR}/ext_roc_curve.png")
plt.close()

# ==== Save prediction CSV (Optional) ====
import pandas as pd
df_out = pd.DataFrame({
    "PREFIX": track_ids[:, 0],
    "TRACK_ID": track_ids[:, 1],
    "True_Label": y_true,
    "Pred_Label": y_pred
})
df_out.to_csv(f"{SEQ_RESULT_DIR}/ext_test_predictions.csv", index=False)
print("✅ Saved test prediction results.")
