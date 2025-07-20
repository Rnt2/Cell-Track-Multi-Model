import os
import pandas as pd
import matplotlib.pyplot as plt

from Config import DATA_DIR, MODEL_DIR, RESULTS_DIR, SEQ_LEN, GENERATED_DIR, HIDDEN_SIZE_LSTM, DROPOUT
from Step1_data import (load_annotations,load_tracks_and_spots,
                        filter_valid_trajectories, compute_features,
                        align_and_save_dataset, build_track_level_dataset)
from Step8_unified_fusion import Train_UnifiedFusionModel
from Step9_SHAP_Unified import SHAP_UnifiedFusionModel

# === features and Config ===
all_features = [  # time-based
    'RADIUS', 'AREA', 'PERIMETER', 'CIRCULARITY',
    'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 'ELLIPSE_ASPECTRATIO', 'SOLIDITY', 'SPEED'
]

all_track_features = [  # track-based
    "TRACK_DURATION", "TRACK_DISPLACEMENT", "TRACK_MEAN_SPEED",
    "TRACK_MAX_SPEED", "TRACK_MIN_SPEED", "TRACK_STD_SPEED",
    "TOTAL_DISTANCE_TRAVELED", "MAX_DISTANCE_TRAVELED", "CONFINEMENT_RATIO",
    "MEAN_STRAIGHT_LINE_SPEED", "LINEARITY_OF_FORWARD_PROGRESSION",
    "MEAN_DIRECTIONAL_CHANGE_RATE"
]

ablation_configs = {
    "baseline": {
        "features": all_features,
        "track_features": all_track_features
    },
    # "remove_negative": {
    #     "features": all_features,
    #     "track_features": [
    #         f for f in all_track_features if f not in [
    #             "TRACK_MEAN_SPEED", "TRACK_STD_SPEED", "MEAN_STRAIGHT_LINE_SPEED"
    #         ]
    #     ]
    # },
    # "top6_shap": {
    #     "features": ["RADIUS", "SOLIDITY", "ELLIPSE_MINOR"],
    #     "track_features": ["TRACK_DURATION", "TRACK_STD_SPEED", "TRACK_MEAN_SPEED"]
    # }
    "Specify" : {
        "features": ['RADIUS', 'AREA', 'PERIMETER', 'CIRCULARITY', 
                     'ELLIPSE_ASPECTRATIO', 'SOLIDITY', 'SPEED'],
        "track_features" :["TRACK_DISPLACEMENT", "TOTAL_DISTANCE_TRAVELED",
                            "MEAN_DIRECTIONAL_CHANGE_RATE"]
    }
}


results_summary = []

# === begin experiment ===
for name, cfg in ablation_configs.items():
    print(f"\n===== Running Ablation: {name} =====")

    # model and dataset save route
    prefix = f"ablation_{name}"
    seq_path = os.path.join(GENERATED_DIR, f"{prefix}_{SEQ_LEN}.npz")
    track_path = os.path.join(GENERATED_DIR, f"{prefix}track_dataset.npz")
    
    model_path = os.path.join(MODEL_DIR, f"{prefix}.pth")
    result_path = os.path.join(RESULTS_DIR, prefix)
    os.makedirs(result_path, exist_ok=True)

    # step 1: create dataset
    cart_labels = load_annotations(f"{DATA_DIR}/CART annotations.xlsx",
                                   is_second_batch=False)
    second_labels = load_annotations(f"{DATA_DIR}/2nd batch annotations.xlsx",
                                     is_second_batch=True)
    
    spots_df, tracks_df = load_tracks_and_spots(
        folder=f"{DATA_DIR}/TRACK",
        cart_labels=cart_labels,
        second_labels=second_labels
    )
    
    spots_df, tracks_df = filter_valid_trajectories(spots_df, tracks_df)
    spots_df = compute_features(spots_df)

    align_and_save_dataset(spots_df,
                            cfg["features"], seq_len=SEQ_LEN,
                            output_prefix=prefix)
    
    build_track_level_dataset(tracks_df, cart_labels, second_labels,
                              prefix, cfg["track_features"])    



    # Step2: training model
    seq_input_size = len(cfg["features"])
    track_input_size = len(cfg["track_features"])

    metrics = Train_UnifiedFusionModel(
        seq_path=seq_path,
        track_path=track_path,
        model_save_path=model_path,
        result_path=result_path,
        seq_input_size=seq_input_size,
        track_input_size=track_input_size,
        hidden_size=HIDDEN_SIZE_LSTM,
        dropout=DROPOUT
    )

    # Step3: shap analysis
    SHAP_UnifiedFusionModel(
        seq_length=SEQ_LEN,
        features=cfg["features"],
        track_features=cfg["track_features"],
        model_save_path=model_path,
        result_path=result_path,
        seq_path=seq_path,
        track_path=track_path
    )

    # Step4: record result
    results_summary.append({
        "config_name": name,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "auc": metrics["auc"]
    })

# === show and save comparison sheet ===
df = pd.DataFrame(results_summary)
df = df.sort_values(by="accuracy", ascending=False)
print("\n=== Summary of Ablation Results ===")
print(df)

summary_path = os.path.join(RESULTS_DIR, f"ablation_summary_{SEQ_LEN}.csv")
df.to_csv(summary_path, index=False)
print(f"Saved summary to {summary_path}")


# === save plot ===
plt.figure(figsize=(10, 5))
df.plot(x='config_name', y=['accuracy', 'f1_score', 'auc'], kind='bar')
plt.title(f"Ablation Performance Comparison (SeqLen={SEQ_LEN})")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"ablation_barplot_{SEQ_LEN}.png"))
plt.close()
print("Saved performance bar plot.")