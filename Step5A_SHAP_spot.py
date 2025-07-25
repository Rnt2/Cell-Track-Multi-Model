import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
from Step3A_spotmodel import BiLSTMAttnModel
from Config import *  
# === Step 1: Load Data ===
data = np.load(f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz")
X, y = data["X"], data["y"]
MODEL_PATH=f"{MODEL_DIR}/Model_best_{SEQ_LEN}.pth"
print("X shape:", X.shape)  # (N, 11, 20)

# fix shape:  (N, 11, 20)-> (N, 20, 11)
if X.shape[1] == 11 and X.shape[2] == 20:
    print("transposing...")
    X = np.transpose(X, (0, 2, 1))  # (N, 20, 11)

print("X shape:", X.shape)  # should be (N, 20, 11)

X = torch.tensor(X, dtype=torch.float32)

# y: long
from sklearn.preprocessing import LabelEncoder
y_encoded = LabelEncoder().fit_transform(y)
y_encoded = torch.tensor(y_encoded, dtype=torch.long)

# === Step 2: Split dataset ===
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === Step 3: Define and load model ===
class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        logits, _ = self.base(x)
        return logits

model = BiLSTMAttnModel(input_dim=FEATURE_LEN, hidden_dim=128, output_dim=3, dropout=0.3)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
wrapped_model = WrappedModel(model)

# === Step 4: Select background and test samples ===
background = X_train[:100]
test_samples = X_test[:100]
# ========== 1. Create SHAP Explainer ==========
explainer = shap.GradientExplainer(wrapped_model, background)
shap_values = explainer.shap_values(test_samples, ranked_outputs=3)

# ========== 2. Explain dimension details ==========
# shap_values[0] shape: (samples, time, features, classes)
shap_tensor = shap_values[0]  # shape = (N, T, F, C)
print("SHAP tensor shape:", shap_tensor.shape)

N, T, F, C = shap_tensor.shape
# 100, 20, 9 , 3


if F != len(features):
    print("Default feature names")
    features = [f"Feature {i}" for i in range(F)]

# ========== 3. Visualize every class ==========
for class_id in range(C):
    shap_class = shap_tensor[:, :, :, class_id]  # shape: (N, T, F)
    mean_shap = np.abs(shap_class).mean(axis=(0, 1))  # shape: (F,)
    
    df = pd.DataFrame({
        "Feature": features,
        "Importance": mean_shap
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(11, 5))
    sns.barplot(data=df, x="Importance", y="Feature")
    plt.title(f"Mean SHAP Feature Importance - Class {class_id}")
    plt.tight_layout()
    plt.savefig(f"{SEQ_RESULT_DIR}/shap_bar_class{class_id}.png")
    plt.close()
    print(f"Saved shap_bar_class{class_id}.png")

# ========== 4. Merge all classes' SHAP values and plot overall ==========
mean_overall = np.abs(shap_tensor).mean(axis=(0, 1, 3))  # shape: (F,)

df_all = pd.DataFrame({
    "Feature": features,
    "Importance": mean_overall
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(11, 5))
sns.barplot(data=df_all, x="Importance", y="Feature")
plt.title("Overall SHAP Feature Importance (All Classes)")
plt.tight_layout()
plt.savefig(f"{SEQ_RESULT_DIR}/shap_bar_overall.png")
plt.close()
print("Saved shap_bar_overall.png")
