import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import random

def load_annotation_dict(annotation_path):
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    print(f"[INFO] Loading annotation from: {annotation_path}")
    ext = os.path.splitext(annotation_path)[-1]

    if ext == ".xlsx":
        excel = pd.ExcelFile(annotation_path)
        sheets = excel.sheet_names
        if "Statistics" in sheets:
            df = pd.read_excel(annotation_path, sheet_name="Statistics")
            df["Device Name"] = df["Name"].astype(str).str.replace(" ", "").str.lower()
            mapping = dict(zip(df["Device Name"], df["Score"]))
        elif "Sheet1" in sheets:
            df = pd.read_excel(annotation_path, sheet_name="Sheet1")
            df["Device Name"] = df["Meso IL18 CAR T cells"].astype(str).str.replace(" ", "").str.lower()
            mapping = dict(zip(df["Device Name"], df["Labels"]))
        elif "Summary" in sheets:
            df = pd.read_excel(annotation_path, sheet_name="Summary")
            df = df.iloc[:, [1, 2]]
            df.columns = ["Name", "Score"]
            df["Device Name"] = df["Name"].astype(str).str.replace(" ", "").str.lower()
            mapping = dict(zip(df["Device Name"], df["Score"]))
        else:
            raise ValueError(f"Unsupported sheet format in: {annotation_path}")
    elif ext == ".csv":
        df = pd.read_csv(annotation_path)
        if "Name" in df.columns and "Score" in df.columns:
            df["Device Name"] = df["Name"].astype(str).str.replace(" ", "").str.lower()
            mapping = dict(zip(df["Device Name"], df["Score"]))
        else:
            raise ValueError("CSV annotation must have 'Name' and 'Score' columns.")
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_path}")

    print(f"[INFO] Total entries loaded: {len(mapping)}")
    for k, v in list(mapping.items())[:5]:
        print(f"  {k} → {v}")
    return mapping

def load_track_spot_pair(data_dir, dataset_name, annotation_dict):
    folder_path = os.path.join(data_dir, dataset_name)
    print(f"\n[INFO] Loading data for dataset: {dataset_name} from {folder_path}")

    all_spots, all_tracks = [], []

    for fname in os.listdir(folder_path):
        if not fname.endswith("_tracks.csv"):
            continue

        prefix = fname.replace("_tracks.csv", "")
        id_part = 1 if prefix.startswith("2nd") else 0
        device_name = prefix.split("_")[id_part].replace(" ", "").lower()

        if device_name not in annotation_dict:
            print(f"[WARNING] Device '{device_name}' not in annotation, skipping.")
            continue

        label = annotation_dict[device_name]
        spot_file = os.path.join(folder_path, prefix + "_spots.csv")
        track_file = os.path.join(folder_path, prefix + "_tracks.csv")

        try:
            df_raw_track = pd.read_csv(track_file, encoding="latin1", header=None)
            names_track = df_raw_track.iloc[0].tolist()
            df_track = pd.read_csv(track_file, encoding="latin1", skiprows=4, names=names_track)
            df_track["PREFIX"] = prefix
            df_track["LABEL"] = label
            df_track["SOURCE"] = dataset_name

            df_raw_spot = pd.read_csv(spot_file, encoding="latin1", header=None)
            names_spot = df_raw_spot.iloc[0].tolist()
            df_spot = pd.read_csv(spot_file, encoding="latin1", skiprows=4, names=names_spot)
            df_spot["PREFIX"] = prefix
            df_spot["LABEL"] = label
            df_spot["SOURCE"] = dataset_name

            all_tracks.append(df_track)
            all_spots.append(df_spot)

        except Exception as e:
            print(f"[ERROR] Failed to load {prefix}: {e}")
            continue

    if len(all_spots) == 0 or len(all_tracks) == 0:
        raise ValueError(f"[ERROR] No valid files found in {dataset_name}")

    spots_df = pd.concat(all_spots, ignore_index=True)
    tracks_df = pd.concat(all_tracks, ignore_index=True)
    print(f"[INFO] Loaded {len(spots_df)} spot rows and {len(tracks_df)} track rows for {dataset_name}")
    return spots_df, tracks_df

def filter_valid_trajectories(spots_df, tracks_df, min_frames=10):
    valid_ids = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames][["PREFIX", "TRACK_ID"]]
    spots_df_filtered = spots_df.merge(valid_ids, on=["PREFIX", "TRACK_ID"], how="inner")
    tracks_df_filtered = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames]
    print(f"[INFO] Filtered spots: {len(spots_df_filtered)} | tracks: {len(tracks_df_filtered)}")
    return spots_df_filtered, tracks_df_filtered

def compute_spot_features(spots_df):
    print("[INFO] Computing spot-level features...")
    spots_df = spots_df.sort_values(by=["PREFIX", "TRACK_ID", "FRAME"])
    spots_df["VELOCITY_X"] = spots_df.groupby(["PREFIX", "TRACK_ID"])["POSITION_X"].diff().fillna(0)
    spots_df["VELOCITY_Y"] = spots_df.groupby(["PREFIX", "TRACK_ID"])["POSITION_Y"].diff().fillna(0)
    spots_df["SPEED"] = np.sqrt(spots_df["VELOCITY_X"]**2 + spots_df["VELOCITY_Y"]**2)
    spots_df["DIRECTION"] = np.arctan2(spots_df["VELOCITY_Y"], spots_df["VELOCITY_X"]) / np.pi
    drop_cols = [col for col in spots_df.columns if "INTENSITY" in col or col in ["POSITION_X", "POSITION_Y"]]
    spots_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    for col in spots_df.columns:
        if col not in ["PREFIX", "TRACK_ID", "FRAME", "LABEL", "SOURCE"]:
            spots_df[col] = pd.to_numeric(spots_df[col], errors="coerce")
    spots_df = spots_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return spots_df

def filter_outlier_spots(spots_df):
    print("[INFO] Filtering outlier spots...")
    init_rows = len(spots_df)
    spots_df = spots_df[
        (spots_df["ELLIPSE_MINOR"] != 0) &
        (spots_df["ELLIPSE_ASPECTRATIO"] > 0) &
        (spots_df["ELLIPSE_ASPECTRATIO"] <= 5)
    ]
    print(f"[INFO] Removed {init_rows - len(spots_df)} invalid rows | Remaining: {len(spots_df)}")
    return spots_df

def save_spot_sequence_dataset(spots_df, feature_columns, generated_dir, seq_len=20, output_prefix="trajectory_dataset", save_csv=True):
    print(f"[INFO] Saving dataset: {output_prefix}_{seq_len}")
    X_list, y_list, track_id_list, rows = [], [], [], []

    for (prefix, group_df) in spots_df.groupby("PREFIX"):
        source = group_df["SOURCE"].iloc[0]
        scaler = StandardScaler()
        scaler.fit(group_df[feature_columns])

        for (p, tid), traj in group_df.groupby(["PREFIX", "TRACK_ID"]):
            feat = traj[feature_columns].copy()
            if len(feat) >= seq_len:
                feat = feat.iloc[:seq_len]
            else:
                pad = pd.DataFrame(np.zeros((seq_len - len(feat), len(feature_columns))), columns=feature_columns)
                feat = pd.concat([feat, pad], ignore_index=True)

            feat_scaled = scaler.transform(feat)
            X_list.append(feat_scaled)
            y_list.append(traj["LABEL"].iloc[0])
            track_id_list.append((p, tid, source))

            if save_csv:
                for t in range(seq_len):
                    rows.append([f"{p}_{tid}", t, source] + list(feat_scaled[t]))

    X = np.array(X_list)
    y = np.array(y_list)
    track_ids = np.array(track_id_list, dtype=object)

    np.savez(os.path.join(generated_dir, f"{output_prefix}_{seq_len}.npz"), X=X, y=y, track_ids=track_ids)
    if save_csv:
        df_out = pd.DataFrame(rows, columns=["SampleID", "Frame", "SOURCE"] + feature_columns)
        df_out.to_csv(os.path.join(generated_dir, f"{output_prefix}_{seq_len}.csv"), index=False)

def create_loso_trials(spots_df, features, generated_dir, seq_len=100, random_seed=42):
    os.makedirs(generated_dir, exist_ok=True)
    random.seed(random_seed)

    label_to_devices = defaultdict(set)
    for prefix in spots_df["PREFIX"].unique():
        label = spots_df[spots_df["PREFIX"] == prefix]["LABEL"].iloc[0]
        source = spots_df[spots_df["PREFIX"] == prefix]["SOURCE"].iloc[0]
        label_to_devices[label].add((source, prefix))

    heldout_devices = []
    print("\n[LOSO] Held-out devices for each label:")
    for label, devices in label_to_devices.items():
        heldout = random.choice(list(devices))
        heldout_devices.append((label, heldout[0], heldout[1]))
        print(f" - Label {label}: {heldout[0]}/{heldout[1]}")

    test_mask = pd.Series(False, index=spots_df.index)
    for label, source, prefix in heldout_devices:
        match = (spots_df["LABEL"] == label) & (spots_df["SOURCE"] == source) & (spots_df["PREFIX"] == prefix)
        test_mask |= match

    test_df = spots_df[test_mask]
    train_df = spots_df[~test_mask]

    postfix = "_".join([f"L{l}_{s}_{p}" for l, s, p in heldout_devices])
    save_spot_sequence_dataset(train_df, features, generated_dir, seq_len, output_prefix=f"loso_train_{postfix}", save_csv=False)
    save_spot_sequence_dataset(test_df, features, generated_dir, seq_len, output_prefix=f"loso_test_{postfix}", save_csv=False)

    print(f"[LOSO] Total train: {len(train_df)} | test: {len(test_df)}")

if __name__ == "__main__":
    from Config import DATA_DIR, GENERATED_DIR, features, SEQ_LEN

    paths = {
        "track": os.path.join(DATA_DIR, "CART_annotation.xlsx"),
        "2nd": os.path.join(DATA_DIR, "2nd_annotation.xlsx"),
        "pdo": os.path.join(DATA_DIR, "PDO_annotation.xlsx")
    }

    all_spots_df = []
    all_tracks_df = []
    label_to_devices = defaultdict(set)
    for dataset_name, anno_path in paths.items():
        print(f"\n=== Processing dataset: {dataset_name.upper()} ===")
        mapping = load_annotation_dict(anno_path)
        spots_df, tracks_df = load_track_spot_pair(DATA_DIR, dataset_name, mapping)
        all_spots_df.append(spots_df)
        all_tracks_df.append(tracks_df)
        for prefix in spots_df["PREFIX"].unique():
            label = spots_df[spots_df["PREFIX"] == prefix]["LABEL"].iloc[0]
            label_to_devices[label].add((dataset_name, prefix))

    merged_spots_df = pd.concat(all_spots_df, ignore_index=True)
    merged_tracks_df = pd.concat(all_tracks_df, ignore_index=True)

    merged_spots_df, merged_tracks_df = filter_valid_trajectories(merged_spots_df, merged_tracks_df)
    merged_spots_df = compute_spot_features(merged_spots_df)
    merged_spots_df = filter_outlier_spots(merged_spots_df)

    save_spot_sequence_dataset(
        merged_spots_df,
        features,
        GENERATED_DIR,
        seq_len=SEQ_LEN,
        output_prefix="trajectory_dataset"
    )

    '''create_loso_trials(
        spots_df=merged_spots_df,
        features=features,
        generated_dir=GENERATED_DIR,
        seq_len=SEQ_LEN
    )'''
