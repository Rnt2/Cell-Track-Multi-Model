import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd
from Config import HIDDEN_SIZE_LSTM, DROPOUT

# === model and variable import ===

from Step8_unified_fusion import UnifiedFusionModel, load_and_align_data
torch.backends.cudnn.enabled = False

# === routes configuration ===

def SHAP_UnifiedFusionModel(seq_length, features,track_features,model_save_path,result_path,seq_path,track_path):
    """
    Perform SHAP analysis on the unified fusion model.
    """
    feature_length = len(features)  # e.g. 9
    track_feature_length = len(track_features)  # e.g. 12
    total_seq = seq_length * feature_length  # e.g. 180 = 20 * 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === load data and model  ===
    print("[STEP 1] Loading model and data...")

    model = UnifiedFusionModel(seq_input_size=feature_length, track_input_size=track_feature_length, hidden_size=HIDDEN_SIZE_LSTM, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device,weights_only=True))
    model.eval()

    X_seq, X_track, y = load_and_align_data(seq_path, track_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_seq_train, X_seq_test, X_track_train, X_track_test, y_train, y_test = train_test_split(
        X_seq, X_track, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32)
    X_track_train = torch.tensor(X_track_train, dtype=torch.float32)
    X_track_test = torch.tensor(X_track_test, dtype=torch.float32)


    # combine (X_seq, X_track) to one tensor
    X_seq_flat = X_seq_test.reshape(X_seq_test.shape[0], -1)  # (N, 20*9)
    X_test_concat = torch.cat([X_seq_flat, X_track_test], dim=1).to(device)

    X_train_seq_flat = X_seq_train.reshape(X_seq_train.shape[0], -1)
    X_train_concat = torch.cat([X_train_seq_flat, X_track_train], dim=1).to(device)
    # define wrapped model
    class WrappedUnifiedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.seq_length = seq_length
            self.feature_length = feature_length

        def forward(self, x_concat):
            # x_concat: (100, 192) 192 = 9*T + 12
            batch_size = x_concat.shape[0]
            x_seq_flat = x_concat[:, :total_seq]  # T=20, F=9
            x_track = x_concat[:, total_seq:]
            x_seq = x_seq_flat.view(batch_size, self.seq_length, self.feature_length)
            return self.model(x_seq, x_track)
    # define shap
    print("[STEP 2] SHAP analysis...")
    wrapped_model = WrappedUnifiedModel(model).to(device)
    explainer = shap.GradientExplainer(wrapped_model, X_train_concat[:100])
    shap_values = explainer.shap_values(X_test_concat[:100], ranked_outputs=1)

    print("[STEP 3] SHAP drawing...")
    # open (100, total_seq+12, 1) 
    # get shap value from package 
    shap_values_combined = shap_values[0]

    shap_value_seq = shap_values_combined[:, :total_seq]
    shap_value_track = shap_values_combined[:, total_seq:]

    shap_value_seq = shap_value_seq.reshape(100, seq_length, feature_length)

    shap_result_seq = shap_value_seq.mean(axis=(0,1))
    shap_result_track = shap_value_track.mean(axis=(0,2))

    feature_names = features + track_features

    shap_result = np.concatenate((shap_result_seq, shap_result_track)) 


    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": shap_result
    }).sort_values("Importance", ascending=False)



    plt.figure(figsize=(12, 6))
    sns.barplot(data=shap_df.head(30), x="Importance", y="Feature")
    plt.title("Unified fusion SHAP Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{result_path}/unified_shap_bar.png")
    print("SHAP feature importance saved.")
    plt.close()


    print("[STEP 4] SHAP summary plot (beeswarm)...")

    # 处理 shap_values 为 2D numpy
    shap_values_2d = shap_values_combined.squeeze(-1)  # shape: (100, 192)

    # 构造 feature names
    seq_feature_names = [f"{feat}_t{t}" for t in range(seq_length) for feat in features]
    feature_names = seq_feature_names + track_features  # length = 192

    X_test_concat_cpu = X_test_concat[:100].cpu().detach().numpy()

    # summary plot（蜂群图）
    shap.summary_plot(
        shap_values_2d,
        X_test_concat_cpu,
        feature_names=feature_names,
        plot_type="dot",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{result_path}/unified_shap_summary.png")
    print("SHAP summary plot saved.")
    plt.close()

if __name__ == "__main__":
    from Config import UNI_RESULT_DIR, SEQ_LEN, MODEL_DIR, SEQ_RESULT_DIR, features, track_features, GENERATED_DIR
     # e.g. 180 = 20 * 9
    SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
    TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"
    MODEL_SAVE_PATH = f"{MODEL_DIR}/unified_fusion_model.pth"
    os.makedirs(SEQ_RESULT_DIR, exist_ok=True)
    SHAP_UnifiedFusionModel(SEQ_LEN, features, track_features, MODEL_SAVE_PATH, result_path=UNI_RESULT_DIR, seq_path=SEQ_DATA_PATH, track_path=TRACK_DATA_PATH) 