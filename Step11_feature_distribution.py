import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for environments without GUI
import matplotlib.pyplot as plt
from Config import GENERATED_DIR, features, track_features
import numpy as np
import scipy.stats as stats

save_dir = os.path.join(GENERATED_DIR, "feature_distribution")
os.makedirs(save_dir, exist_ok=True)


def plot_feature_distribution_by_label(feature_name, sem_table, scaler=False):
    """
    Plot the mean and standard deviation of a given feature grouped by label.

    Args:
        feature_name (str): The name of the feature to plot. Should be in either features or track_features.
    """

    # Check if the feature is valid and determine its data level
    valid_features = track_features + features
    is_track_level = feature_name in track_features
    if feature_name not in valid_features:
        raise ValueError(f"Invalid feature name: {feature_name}. Must be one of: {valid_features}")

    # Load unnormalized data
    if scaler:
        csv_file = "track_dataset.csv" if is_track_level else "trajectory_dataset_20.csv"
    else:
        csv_file = "unscaled_track_features.csv" if is_track_level else "unscaled_spot_features.csv"
    df = pd.read_csv(os.path.join(GENERATED_DIR, csv_file))

    if feature_name not in df.columns or 'LABEL' not in df.columns:
        raise ValueError(f"'{feature_name}' or 'LABEL' not found in dataframe.")

    # Compute mean and standard deviation for each label
    stats = df.groupby("LABEL")[feature_name].agg(['mean', 'std', 'count']).reset_index()
    stats['sem'] = stats['std'] / stats['count']**0.5
    
    for i, row in stats.iterrows():
        sem_table.append({
            "feature": feature_name,
            "label": row["LABEL"],
            "sem": row["sem"]
        })
    
    # Create bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(stats["LABEL"], stats["mean"], yerr=stats["sem"],
            width=0.5, capsize=8, alpha=0.7, color='skyblue')
    plt.xlabel("Label")
    plt.ylabel(f"{feature_name} Mean Value")
    plt.title(f"{feature_name} Mean ± SEM by Label")
    plt.xticks(stats["LABEL"])

    # Annotate each bar with its value
    for i, row in stats.iterrows():
        plt.text(row["LABEL"], row["mean"] + row["sem"] * 0.05, f'{row["mean"]:.2f}', 
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save plot to file
    out_path = os.path.join(save_dir, f"feature_{feature_name}_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")



# two sample mean z-test, save as Excel file
def z_test_pairwise(features_list):
    # current only spot
    csv_file = "unscaled_spot_features.csv"
    df = pd.read_csv(os.path.join(GENERATED_DIR, csv_file))
    # groups = [0.0, 0.5, 1.0]
    pairs = [(0.0, 0.5), (0.5, 1.0), (0.0, 1.0)]
    results = []
    for feature in features_list:
        row = {"feature": feature}
        for g1, g2 in pairs:
            d1 = df[df["LABEL"] == g1][feature].dropna()
            d2 = df[df["LABEL"] == g2][feature].dropna()
            n1, n2 = len(d1), len(d2)
            if n1 < 2 or n2 < 2:
                p = np.nan
            else:
                mean1, mean2 = d1.mean(), d2.mean()
                std1, std2 = d1.std(ddof=1), d2.std(ddof=1)
                denom = np.sqrt(std1**2/n1 + std2**2/n2)
                if denom == 0:
                    p = np.nan
                else:
                    z = (mean1 - mean2) / denom
                    p = 2 * stats.norm.sf(abs(z))
            # 4 digits
            if pd.isna(p):
                p_str = ''
            else:
                p_str = f"{p:.4e}"
            row[f"p_{g1}_vs_{g2}"] = p_str
        results.append(row)
    df_out = pd.DataFrame(results)
    out_path = os.path.join(save_dir, "feature_pairwise_ztest_pvalues.xlsx")
    df_out.to_excel(out_path, index=False)
    print(f"Saved z-test p-values to {out_path}")

    
# Example usage
if __name__ == "__main__":
    sem_table = []
    for feature in features + track_features:
        plot_feature_distribution_by_label(feature, sem_table)
    # save SEM table
    sem_df = pd.DataFrame(sem_table)
    sem_out_path = os.path.join(save_dir, "feature_SEM_table.xlsx")
    sem_df.to_excel(sem_out_path, index=False)
    print(f"Saved SEM table to {sem_out_path}")
    
    z_test_pairwise(features)