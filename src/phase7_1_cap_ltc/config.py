"""
Global configuration for Phase 7 — CAP-LTC
------------------------------------------
Single-source-of-truth (variable-channel + uncertainty aware)
Journal- and PhD-safe configuration
"""

from pathlib import Path
import torch

# =============================================================================
# PROJECT ROOTS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

PHASE2_FIXED_DIR = RESULTS_DIR / "phase2_features_fixed"
PREPARED_DIR = DATA_DIR / "processed"

# =============================================================================
# HARDWARE
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # Windows-safe

# =============================================================================
# DATA SHAPE
# =============================================================================
FEATURE_DIM = 128
WINDOW_SIZE = 30

# =============================================================================
# PREPROCESSING
# =============================================================================
NORMALIZE_FEATURES = True

# =============================================================================
# WINDOWING
# =============================================================================
OVERLAPS = [0.0, 0.5, 0.75]   # used for automated experiments
OVERLAP_PERCENT = 0.75       # default/main experiment

# =============================================================================
# TRAINING
# =============================================================================
TRAIN_NORMAL_ONLY = True    
 # one-class learning (core novelty)
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 25             # upper bound (early stopping will stop earlier)
SEED = 42                    # fixed for reproducibility (reported in paper)

# Multi-seed reproducibility (used by `multi_seed.py`)
SEEDS = [42, 49, 56]

# =============================================================================
# EARLY STOPPING (CRITICAL FOR CPU TRAINING)
# =============================================================================
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3  # stop if no improvement for 5 epochs
MIN_DELTA = 1e-4             # minimum loss improvement

# =============================================================================
# MODEL
# =============================================================================
LTC_HIDDEN = 128
ENCODER_HIDDEN = 256

# =============================================================================
# CAP STATES (CRITICAL — UNCERTAINTY AWARE)
# =============================================================================
CAP_MAPPING = {
    "NC": 0,
    "Non-CAP": 0,
    "A1": 1,
    "A2": 2,
    "A3": 3,
    "A?": 4,     # explicit uncertainty state
}

N_CAP_STATES = 5

# =============================================================================
# CAP-LTC DYNAMICS
# =============================================================================
DT = 1.0

TAU_BASE_INIT = 0.5
TAU_CAP_INIT = [
    0.0,   # Non-CAP
    0.5,   # A1
    0.2,   # A2
   -0.5,   # A3
    0.1,   # A? (uncertain CAP)
]

# Numerical stability bounds for τ(t) after softplus.
# Keep these aligned with `src/config/global_config.yaml` if used elsewhere.
TAU_MIN = 0.05
TAU_MAX = 10.0

# =============================================================================
# INFERENCE / THRESHOLDING
# =============================================================================
THRESHOLD_PERCENTILE = 95    # learned from NORMAL windows only

# =============================================================================
# OUTPUTS
# =============================================================================
#CHECKPOINT_NAME = "cap_ltc_best.pt"

MODEL_DIR = RESULTS_DIR / "models"
TAU_DIR = RESULTS_DIR / "tau"
LOG_DIR = RESULTS_DIR / "logs"
FIGURE_DIR = RESULTS_DIR / "figures"

for d in [MODEL_DIR, TAU_DIR, LOG_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SUMMARY (for logs & debugging)
# =============================================================================
def summary():
    print("\n" + "=" * 80)
    print("PHASE 7 CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Device               : {DEVICE}")
    print(f"Feature Dim          : {FEATURE_DIM}")
    print(f"Window Size          : {WINDOW_SIZE}")
    print(f"Normalize Features   : {NORMALIZE_FEATURES}")
    print(f"Overlaps             : {OVERLAPS}")
    print(f"Training Overlap     : {OVERLAP_PERCENT}")
    print(f"Train Normal Only    : {TRAIN_NORMAL_ONLY}")
    print(f"Batch Size           : {BATCH_SIZE}")
    print(f"Learning Rate        : {LEARNING_RATE}")
    print(f"Max Epochs           : {MAX_EPOCHS}")
    print(f"Early Stopping       : {EARLY_STOPPING}")
    print(f"Early Stop Patience  : {EARLY_STOPPING_PATIENCE}")
    print(f"CAP States           : {CAP_MAPPING}")
    print(f"LTC Hidden           : {LTC_HIDDEN}")
    print(f"Encoder Hidden       : {ENCODER_HIDDEN}")
    print(f"DT                   : {DT}")
    print(f"TAU Base Init        : {TAU_BASE_INIT}")
    print(f"TAU CAP Init         : {TAU_CAP_INIT}")
    print(f"Threshold Percentile : {THRESHOLD_PERCENTILE}")
    print("=" * 80 + "\n")
