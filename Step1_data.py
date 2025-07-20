import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Config import DATA_DIR, GENERATED_DIR, features, track_features,SEQ_LEN
# def Create_Dataset(DATA_DIR, GENERATED_DIR, features, track_features,output_prefix,default_seq_len=[SEQ_LEN]):
def save_unscaled_spot_features(spots_df, output_prefix="unscaled_spot_features"):
    # extract unscaled features and save to CSV
    unscaled_df = spots_df[["PREFIX", "TRACK_ID", "FRAME", "LABEL"] + features].copy()
    out_path = os.path.join(GENERATED_DIR, f"{output_prefix}.csv")
    unscaled_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved unscaled spot features to: {out_path}")

def save_unscaled_track_features(tracks_df, output_prefix="unscaled_track_features"):
    # extract unscaled track features and save to CSV
    def match_label(prefix):
        if prefix.startswith("2nd_"):
            prefix_base = prefix.replace("2nd_", "").split("_")[0]
            return second_labels.get(prefix_base, np.nan)
        else:
            prefix_base = prefix.split("_")[0]
            return cart_labels.get(prefix_base, np.nan)

    tracks_df["LABEL"] = tracks_df["PREFIX"].apply(match_label)
    unscaled_df = tracks_df[["PREFIX", "TRACK_ID", "LABEL"] + track_features].copy()
    out_path = os.path.join(GENERATED_DIR, f"{output_prefix}.csv")
    unscaled_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved unscaled track features to: {out_path}")




# === Step 1: Load Annotations ===
def load_annotations(path, is_second_batch=False):
    if is_second_batch:
        df = pd.read_excel(path, sheet_name=0)
        id_series = df["Meso IL18 CAR T cells"].astype(str).str.strip()
        label_series = df["Labels"].astype(float)
    else:
        df = pd.read_excel(path, sheet_name="Summary")
        id_series = df.iloc[:, 1].astype(str).str.strip()
        label_series = df.iloc[:, 2].astype(float)

    mapping = dict(zip(id_series, label_series))
    print(f"Loaded annotation from {path}")
    print("Total entries:", len(mapping))
    return mapping


# === Step 2: Load Track/Spot Files ===
def load_tracks_and_spots(folder, cart_labels, second_labels):
    all_spots = []
    all_tracks = []

    for file_name in os.listdir(folder):
        if not file_name.endswith("_tracks.csv"):
            continue
        
        prefix = file_name.replace("_tracks.csv", "")
        spot_file = file_name.replace("_tracks.csv", "_spots.csv")

        track_path = os.path.join(folder, file_name)
        spot_path = os.path.join(folder, spot_file)

        if prefix.startswith("2nd_"):
            prefix_base = prefix.replace("2nd_", "").split("_")[0]
            label_dict = second_labels
        else:
            prefix_base = prefix.split("_")[0]
            label_dict = cart_labels

        if prefix_base not in label_dict:
            print(f"{prefix_base} cannot be found in annotation")
            continue

        try:
            df_raw_track = pd.read_csv(track_path, encoding='latin1',
                                       header=None)
            names_track = df_raw_track.iloc[0].tolist()
            df_track = pd.read_csv(track_path, encoding='latin1',
                                   skiprows=4, names=names_track)
            df_track['PREFIX'] = prefix

            df_raw_spot = pd.read_csv(spot_path, encoding='latin1',
                                      header=None)
            names_spot = df_raw_spot.iloc[0].tolist()
            df_spot = pd.read_csv(spot_path, encoding='latin1',
                                  skiprows=4, names=names_spot)
            df_spot['PREFIX'] = prefix
            df_spot['LABEL'] = label_dict[prefix_base]

            all_tracks.append(df_track)
            all_spots.append(df_spot)

        except Exception as e:
            print(f"Failed to load {prefix}: {e}")
            continue

    spots_df = pd.concat(all_spots, ignore_index=True)
    tracks_df = pd.concat(all_tracks, ignore_index=True)
    return spots_df, tracks_df


# === Step 3: Filter Valid Trajectories ===
def filter_valid_trajectories(spots_df, tracks_df, min_frames=10):
    
    valid_ids = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames
                          ][["PREFIX", "TRACK_ID"]]
    spots_df_filtered = spots_df.merge(valid_ids, 
                                       on=["PREFIX", "TRACK_ID"], how="inner")
    tracks_df_filtered = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames]


    return spots_df_filtered, tracks_df_filtered


# === Step 4: Compute Features ===
def compute_features(spots_df):
    spots_df = spots_df.sort_values(by=["PREFIX", "TRACK_ID", "FRAME"])
    spots_df["VELOCITY_X"] = spots_df.groupby(
        ["PREFIX", "TRACK_ID"])["POSITION_X"].diff().fillna(0)
    spots_df["VELOCITY_Y"] = spots_df.groupby(
        ["PREFIX", "TRACK_ID"])["POSITION_Y"].diff().fillna(0)
    
    spots_df["SPEED"] = np.sqrt(spots_df["VELOCITY_X"]**2
                                + spots_df["VELOCITY_Y"]**2)
    spots_df["DIRECTION"] = np.arctan2(spots_df["VELOCITY_Y"],
                                       spots_df["VELOCITY_X"]) / np.pi
    drop_cols = [col for col in spots_df.columns 
                 if "INTENSITY" in col or col in ["POSITION_X", "POSITION_Y"]]
    spots_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    for f in features:
        spots_df[f] = pd.to_numeric(spots_df[f], errors='coerce')
    spots_df = spots_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return spots_df

    

# === Step 5: Align and Save Sequences ===
def align_and_save_dataset(spots_df, features, seq_len=20,
                           output_prefix="trajectory_dataset"):

    # global standardization
    X_list, y_list, track_id_list = [], [], []
    rows = []
    for prefix, group_df in spots_df.groupby("PREFIX"):
        # features_except_dir = [f for f in features if f != "DIRECTION"]
        scaler = StandardScaler()
        # scaler.fit(group_df[features_except_dir])
        scaler.fit(group_df[features])
        for (p, tid), traj in group_df.groupby(["PREFIX", "TRACK_ID"]):
            feat = traj[features].values
            if len(feat) >= seq_len:
                feat = feat[:seq_len]
            else:
                pad = np.zeros((seq_len - len(feat), len(features)))
                feat = np.vstack([feat, pad])

            feat_scaled = feat.copy()
            df_temp = pd.DataFrame(feat, columns=features)
            feat_scaled = scaler.transform(df_temp)

            X_list.append(feat_scaled)
            y_list.append(traj["LABEL"].iloc[0])
            track_id_list.append((p, tid))

            for t in range(seq_len):
                row = [f"{p}_{tid}", t] + list(feat_scaled[t])
                rows.append(row)
    X = np.array(X_list)
    y = np.array(y_list)
    track_ids = np.array(track_id_list, dtype=object)
    np.savez(f"{GENERATED_DIR}/{output_prefix}_{seq_len}.npz",
             X=X, y=y, track_ids=track_ids)

    df_out = pd.DataFrame(rows, columns=["SampleID", "Frame"] + features)
    df_out.to_csv(f"{GENERATED_DIR}/{output_prefix}_{seq_len}.csv", index=False)

    print(
        f"[Save] Dataset saved: {GENERATED_DIR}/{output_prefix}_{seq_len}.npz & .csv "
        f"| Shape: {X.shape}"
    )



# === Step 6: Save Track-Level Dataset ===
def build_track_level_dataset(tracks_df, cart_labels, second_labels,
                              output_prefix1="", 
                              track_features = track_features):
    if len(track_features) == 0:
        print("[INFO] No track features available.")
        return

    def match_label(prefix):
        if prefix.startswith("2nd_"):
            prefix_base = prefix.replace("2nd_", "").split("_")[0]
            return second_labels.get(prefix_base, np.nan)
        else:
            prefix_base = prefix.split("_")[0]
            return cart_labels.get(prefix_base, np.nan)

    tracks_df["LABEL"] = tracks_df["PREFIX"].apply(match_label)
    df = tracks_df.dropna(subset=track_features +
                          ["LABEL", "PREFIX", "TRACK_ID"]).copy()

    records = []
    for prefix, group in df.groupby("PREFIX"):
        scaler = StandardScaler()
        group_feat = group[track_features].values
        group_scaled = scaler.fit_transform(group_feat)

        for i, row in enumerate(group.itertuples()):
            record = {
                "PREFIX": prefix,
                "TRACK_ID": row.TRACK_ID,
                "LABEL": row.LABEL
            }
            for j, f in enumerate(track_features):
                record[f] = group_scaled[i][j]
            records.append(record)

    df_final = pd.DataFrame(records)
    df_final.to_csv(f"{GENERATED_DIR}/{output_prefix1}track_dataset.csv",
                    index=False)

    np.savez(f"{GENERATED_DIR}/{output_prefix1}track_dataset.npz", 
            X=df_final[track_features].values, 
            y=df_final["LABEL"].values,
            track_ids=df_final[["PREFIX", "TRACK_ID"]].values)
    print(f"Saved: {GENERATED_DIR}/{output_prefix1}track_dataset.csv & .npz")


def filter_outer(spots_df):
    init_rows = len(spots_df)
    minor_zero_count = (spots_df['ELLIPSE_MINOR'] == 0).sum()
    aspect_neg_or_zero = (spots_df['ELLIPSE_ASPECTRATIO'] <= 0).sum()
    aspect_too_large = (spots_df['ELLIPSE_ASPECTRATIO'] > 5).sum()

    print(f"[Debug] Total rows before filtering: {init_rows}")
    print(f"[Debug] Rows with ELLIPSE_MINOR == 0: {minor_zero_count}")
    print(f"[Debug] Rows with ELLIPSE_ASPECTRATIO <= 0: {aspect_neg_or_zero}")
    print(f"[Debug] Rows with ELLIPSE_ASPECTRATIO > 5: {aspect_too_large}")

    # filter out invalid rows
    spots_df = spots_df[
        (spots_df['ELLIPSE_MINOR'] != 0) &
        (spots_df['ELLIPSE_ASPECTRATIO'] > 0) &
        (spots_df['ELLIPSE_ASPECTRATIO'] <= 5)
    ]
    minor_zero_count = (spots_df['ELLIPSE_MINOR'] == 0).sum()
    aspect_neg_or_zero = (spots_df['ELLIPSE_ASPECTRATIO'] <= 0).sum()
    aspect_too_large = (spots_df['ELLIPSE_ASPECTRATIO'] > 5).sum()

    print(f"[Debug] Total rows before filtering: {init_rows}")
    print(f"[Debug] Rows with ELLIPSE_MINOR == 0: {minor_zero_count}")
    print(f"[Debug] Rows with ELLIPSE_ASPECTRATIO <= 0: {aspect_neg_or_zero}")
    print(f"[Debug] Rows with ELLIPSE_ASPECTRATIO > 5: {aspect_too_large}")
    filtered_rows = len(spots_df)
    removed_count = init_rows - filtered_rows
    print(f"[Filter] Removed {removed_count} invalid rows | Remaining: {filtered_rows}")
    return spots_df

def create_test_dataset(folder_path, annotation_path,
                        output_prefix="test", seq_len=20, min_frames=10):

    from Config import features, track_features, GENERATED_DIR

    # === 1. Load annotations ===
    anno_df = pd.read_excel(annotation_path,sheet_name="Statistics")
    print(anno_df.head())
    anno_df["Device Name"] = anno_df["Name"].astype(str).str.replace(" ", "").str.lower()
    device_to_score = dict(zip(anno_df["Device Name"], anno_df["Score"]))
    
    all_spots, all_tracks = [], []

    # === 2. Load files ===
    for fname in os.listdir(folder_path):
        if "_tracks.csv" in fname:
            prefix = fname.replace("_tracks.csv", "")
            device_name = prefix.split("_")[0].replace(" ", "").lower()
            if device_name not in device_to_score:
                print(f"[Warning] Device '{device_name}' not in annotation.")
                continue

            label = device_to_score[device_name]
            track_path = os.path.join(folder_path, fname)
            spot_path = os.path.join(folder_path, f"{prefix}_spots.csv")

            try:
                # load track
                df_raw_track = pd.read_csv(track_path, encoding='latin1', header=None)
                names_track = df_raw_track.iloc[0].tolist()
                df_track = pd.read_csv(track_path, encoding='latin1', skiprows=4, names=names_track)
                df_track["PREFIX"] = prefix
                df_track["LABEL"] = label
                all_tracks.append(df_track)

                # load spot
                df_raw_spot = pd.read_csv(spot_path, encoding='latin1', header=None)
                names_spot = df_raw_spot.iloc[0].tolist()
                df_spot = pd.read_csv(spot_path, encoding='latin1', skiprows=4, names=names_spot)
                df_spot["PREFIX"] = prefix
                df_spot["LABEL"] = label
                all_spots.append(df_spot)
            except Exception as e:
                print(f"[Error] {prefix}: {e}")

    if len(all_spots) == 0 or len(all_tracks) == 0:
        print("[Error] No valid files loaded.")
        return

    spots_df = pd.concat(all_spots, ignore_index=True)
    tracks_df = pd.concat(all_tracks, ignore_index=True)

    print(f"[Info] Loaded: {len(spots_df)} spot rows | {len(tracks_df)} track rows")

    # === 3. Filter valid tracks (min_frames) ===
    valid_ids = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames][["PREFIX", "TRACK_ID"]]
    spots_df = spots_df.merge(valid_ids, on=["PREFIX", "TRACK_ID"], how="inner")
    tracks_df = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames]

    # === 4. Compute Spot features ===
    spots_df = spots_df.sort_values(by=["PREFIX", "TRACK_ID", "FRAME"])
    spots_df["VELOCITY_X"] = spots_df.groupby(["PREFIX", "TRACK_ID"])["POSITION_X"].diff().fillna(0)
    spots_df["VELOCITY_Y"] = spots_df.groupby(["PREFIX", "TRACK_ID"])["POSITION_Y"].diff().fillna(0)
    spots_df["SPEED"] = np.sqrt(spots_df["VELOCITY_X"]**2 + spots_df["VELOCITY_Y"]**2)
    spots_df["DIRECTION"] = np.arctan2(spots_df["VELOCITY_Y"], spots_df["VELOCITY_X"]) / np.pi

    drop_cols = [col for col in spots_df.columns if "INTENSITY" in col or col in ["POSITION_X", "POSITION_Y"]]
    spots_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    for f in features:
        spots_df[f] = pd.to_numeric(spots_df[f], errors='coerce')
    spots_df = spots_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # === 5. Save Spot sequences ===
    X_list, y_list, track_id_list, rows = [], [], [], []
    for prefix, group_df in spots_df.groupby("PREFIX"):
        scaler = StandardScaler()
        scaler.fit(group_df[features])

        for (p, tid), traj in group_df.groupby(["PREFIX", "TRACK_ID"]):
            feat = traj[features].copy()
            if len(feat) >= seq_len:
                feat = feat.iloc[:seq_len]
            else:
                pad = pd.DataFrame(np.zeros((seq_len - len(feat), len(features))),
                                columns=features)
                feat = pd.concat([feat, pad], ignore_index=True)
            feat_scaled = scaler.transform(feat)
            X_list.append(feat_scaled)
            y_list.append(traj["LABEL"].iloc[0])
            track_id_list.append((p, tid))

            for t in range(seq_len):
                rows.append([f"{p}_{tid}", t] + list(feat_scaled[t]))

    X = np.array(X_list)
    y = np.array(y_list)
    track_ids = np.array(track_id_list, dtype=object)

    df_seq = pd.DataFrame(rows, columns=["SampleID", "Frame"] + features)
    df_seq.to_csv(f"{GENERATED_DIR}/{output_prefix}_trajectory_{seq_len}.csv", index=False)
    np.savez(f"{GENERATED_DIR}/{output_prefix}_trajectory_{seq_len}.npz", X=X, y=y, track_ids=track_ids)

    print(f"[Save] Trajectory-level test set saved | Shape: {X.shape}")

    # === 6. Save Track-level features ===
    records = []
    for prefix, group in tracks_df.groupby("PREFIX"):
        scaler = StandardScaler()
        
        group_feat = group[track_features].dropna().values
        group_scaled = scaler.fit_transform(group_feat)

        for i, row in enumerate(group.itertuples()):
            record = {
                "PREFIX": prefix,
                "TRACK_ID": row.TRACK_ID,
                "LABEL": row.LABEL
            }
            for j, f in enumerate(track_features):
                record[f] = group_scaled[i][j]
            records.append(record)

    df_track = pd.DataFrame(records)
    df_track.to_csv(f"{GENERATED_DIR}/{output_prefix}_track.csv", index=False)
    np.savez(f"{GENERATED_DIR}/{output_prefix}_track.npz",
             X=df_track[track_features].values,
             y=df_track["LABEL"].values,
             track_ids=df_track[["PREFIX", "TRACK_ID"]].values)
    print(f"[Save] Track-level test set saved | Shape: {df_track.shape}")

    # === 7. Save unscaled version ===
    spots_df.to_csv(f"{GENERATED_DIR}/{output_prefix}_unscaled_spot_features.csv", index=False)
    tracks_df.to_csv(f"{GENERATED_DIR}/{output_prefix}_unscaled_track_features.csv", index=False)
    print(f"[Save] Raw features saved.")

if __name__ == "__main__":

    cart_labels = load_annotations(f"{DATA_DIR}/CART annotations.xlsx",
                                   is_second_batch=False)
    second_labels = load_annotations(f"{DATA_DIR}/2nd batch annotations.xlsx",
                                     is_second_batch=True)
    
    spots_df, tracks_df = load_tracks_and_spots(
        folder=f"{DATA_DIR}/TRACK",
        cart_labels=cart_labels,
        second_labels=second_labels
    )
    # Create train dataset
    spots_df, tracks_df = filter_valid_trajectories(spots_df, tracks_df)
    spots_df = compute_features(spots_df)
    spots_df = filter_outer(spots_df)
    from Config import SEQ_LEN
    for seq_len_iter in [SEQ_LEN]:
        align_and_save_dataset(spots_df,
                               features, seq_len=seq_len_iter,
                               output_prefix="trajectory_dataset")
    
    build_track_level_dataset(tracks_df, cart_labels, second_labels)   

    save_unscaled_spot_features(spots_df)
    save_unscaled_track_features(tracks_df)
    
    print("Dataset creation completed.")
    # Create test dataset
    create_test_dataset(
    folder_path=f"{DATA_DIR}/NEW",
    annotation_path=f"{DATA_DIR}/PDO size change statistics_20250718.xlsx",
    output_prefix="cart_test",
    seq_len=SEQ_LEN
)
