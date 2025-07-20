
import os
# ======= RUNNING SETTINGS =======
SEQ_LEN = 100 # Choose from 20, 100, 360
HIDDEN_SIZE_LSTM = 128
DROPOUT = 0.3
EPOCHS = 400
# ======= PATH =======

# Upload this folder with original data files
DATA_DIR = "./Data"
os.makedirs(DATA_DIR, exist_ok=True)
# Manually delete this folder before uploading
GENERATED_DIR = "./Generated"
os.makedirs(GENERATED_DIR, exist_ok=True)
MODEL_DIR = os.path.join(GENERATED_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
# Upload this folder with results and plots
RESULTS_DIR = "./Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
TRACK_RESULT_DIR = f"{RESULTS_DIR}/track"
os.makedirs(TRACK_RESULT_DIR, exist_ok=True)
SEQ_RESULT_DIR = f"{RESULTS_DIR}/{SEQ_LEN}"
os.makedirs(SEQ_RESULT_DIR, exist_ok=True)
FUSION_RESULT_DIR = f"{RESULTS_DIR}/Fusion_{SEQ_LEN}"
os.makedirs(FUSION_RESULT_DIR, exist_ok=True)
UNI_RESULT_DIR = f"{RESULTS_DIR}/Unified_{SEQ_LEN}"
os.makedirs(UNI_RESULT_DIR, exist_ok=True)
# ======= FEATURES =======

features = [ # Time-based Features 
    # Morphological Features
    'RADIUS', 'AREA', 'PERIMETER', 'CIRCULARITY', # Geometric Properties
    'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 'ELLIPSE_ASPECTRATIO', # Ellipse-fitting-based features
    'SOLIDITY',
    # Motion Features
    'SPEED'  # Calculated 
]

track_features = [ # Track-Level Statistics Features
    "TRACK_DURATION", "TRACK_DISPLACEMENT", "TRACK_MEAN_SPEED",
    "TRACK_MAX_SPEED", "TRACK_MIN_SPEED", "TRACK_STD_SPEED",
    "TOTAL_DISTANCE_TRAVELED", "MAX_DISTANCE_TRAVELED", "CONFINEMENT_RATIO",
    "MEAN_STRAIGHT_LINE_SPEED", "LINEARITY_OF_FORWARD_PROGRESSION",
    "MEAN_DIRECTIONAL_CHANGE_RATE"
]

# ======= CONSTANTS =======
FEATURE_LEN = len(features) # = 9
TRACK_LEN = len(track_features) # = 12

# Run Step1 if you haven't yet to generate the dataset, or any changes to the dataset.
# Run Step2 once to train the track model.
# Change SEQ_LEN to 20, 100, and 360, and run Step 3, 4 and 5 6 under each SEQ_LEN.
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
