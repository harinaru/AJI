"""
Global configuration for Phase 7 — CAP-LTC
------------------------------------------
Single-source-of-truth (variable-channel + uncertainty aware)
Journal- and PhD-safe configuration

CHANGES FROM PREVIOUS VERSION
------------------------------
1. TAU_BASE_INIT: 0.5 → 3.0
   Old value gave γ=DT/τ≈1.0 (no temporal memory).
   New value gives γ≈0.32 (cell integrates ~3 timesteps of context).

2. TAU_CAP_INIT: small offsets → larger offsets [0, 2, 1, -1.5, 0.5]
   Larger offsets create genuinely different γ per CAP state:
     NC/Non-CAP : τ≈3.1, γ≈0.32  (baseline integration)
     A1         : τ≈5.2, γ≈0.19  (slow — sustained attention)
     A2         : τ≈4.1, γ≈0.24  (moderate)
     A3         : τ≈1.7, γ≈0.59  (fast — high-frequency arousal)
     A?         : τ≈3.6, γ≈0.28  (uncertain, near baseline)

3. MAX_EPOCHS: 25 → 75
   25 epochs was never enough. With cosine annealing the model needs
   more time to find the anomaly-discriminative manifold.

4. EARLY_STOPPING_PATIENCE: 3 → 10
   Patience of 3 caused premature stopping before useful learning happened.

5. MARGIN_LOSS_WEIGHT: new — controls anomaly margin training strength
6. MARGIN_LOSS_MARGIN: new — minimum required error gap between normal/disorder
7. LATENT_REG_WEIGHT: unchanged at 0.05 but now explicitly named
8. LEARNING_RATE: 1e-3 (unchanged — good starting value)
9. LR_SCHEDULER: cosine_annealing replaces ReduceLROnPlateau
   Cosine annealing avoids the plateau-detection delay that was causing
   unnecessary early LR decay.
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
OVERLAPS = [0.75]   # used for automated experiments
OVERLAP_PERCENT = 0.75               # default/main experiment

# =============================================================================
# TRAINING
# =============================================================================
TRAIN_NORMAL_ONLY = True    # one-class learning (core novelty)
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Increased from 25 → 75.
# 25 epochs was insufficient for the model to learn anomaly-discriminative
# features. Cosine annealing over 75 epochs gives a much better loss surface.
MAX_EPOCHS = 50

SEED = 42                    # fixed for reproducibility
SEEDS = [42, 49, 56]         # multi-seed reproducibility

# =============================================================================
# EARLY STOPPING
# =============================================================================
EARLY_STOPPING = True

# Increased from 3 → 10.
# Patience of 3 stopped training before any useful anomaly features were found.
EARLY_STOPPING_PATIENCE = 8

MIN_DELTA = 1e-4             # minimum loss improvement

# =============================================================================
# LR SCHEDULER
# =============================================================================
# Using cosine annealing instead of ReduceLROnPlateau.
# Cosine annealing decays smoothly over training, avoiding the stagnation
# detection delay that caused premature LR decay in the old setup.
LR_SCHEDULER = "cosine"          # "cosine" or "plateau"
LR_MIN = 1e-6                    # minimum LR for cosine annealing

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

# --- CHANGED ---
# TAU_BASE_INIT: 0.5 → 3.0
# With old value 0.5: softplus(0.5)≈0.97, τ≈1.02, γ=DT/τ≈0.98 → near-full
# forgetting every step (functionally a feedforward net, not a recurrent one).
# With new value 3.0: softplus(3.0)≈3.05, τ≈3.1, γ≈0.32 → integrates
# ~3 timesteps of context, giving the LTC cell genuine temporal memory.
TAU_BASE_INIT = 3.0

# --- CHANGED ---
# TAU_CAP_INIT: small offsets → larger, physiologically motivated offsets.
# These offsets are added to tau_base BEFORE softplus, creating distinct τ
# values per CAP state. Larger differences = more discriminative temporal
# dynamics between normal (NC/Non-CAP) and arousal (A1/A2/A3) states.
#
# Resulting approximate τ values (τ = TAU_MIN + softplus(TAU_BASE + offset)):
#   NC/Non-CAP : offset=0.0  → τ≈3.10, γ≈0.32
#   A1         : offset=2.0  → τ≈5.15, γ≈0.19  (slow integration)
#   A2         : offset=1.0  → τ≈4.10, γ≈0.24
#   A3         : offset=-1.5 → τ≈1.68, γ≈0.60  (fast integration)
#   A? (uncert): offset=0.5  → τ≈3.63, γ≈0.28
TAU_CAP_INIT = [
    0.0,    # Non-CAP / NC     — baseline
    2.0,    # A1               — slow, sustained (was 0.5)
    1.0,    # A2               — moderate (was 0.2)
   -1.5,    # A3               — fast arousal (was -0.5)
    0.5,    # A? (uncertain)   — near baseline (was 0.1)
]

# Numerical stability bounds for τ(t) after softplus.
TAU_MIN = 0.05
TAU_MAX = 10.0

# =============================================================================
# LOSS WEIGHTS
# =============================================================================

# Latent temporal regularization (smoothness penalty on hidden state changes).
# Unchanged from original.
LATENT_REG_WEIGHT = 0.05

# Anomaly margin loss — pushes disorder reconstruction error above normal error.
# Applied when disorder samples appear in the same batch as normals.
# margin = minimum required error gap between disorder and normal windows.
# weight = how strongly to enforce the margin in the total loss.
MARGIN_LOSS_WEIGHT = 0.15
MARGIN_LOSS_MARGIN = 0.15

# =============================================================================
# INFERENCE / THRESHOLDING
# =============================================================================
THRESHOLD_PERCENTILE = 95    # learned from NORMAL windows only

# --- NEW ---
# Subject score for anomaly detection classification.
# "p90_error" uses the 90th percentile window reconstruction error per subject
# instead of the old mean (mean collapses high-anomaly bursts into noise).
# Options: "p90_error", "p95_error", "instability_index", "max_error"
SUBJECT_SCORE_FIELD = "p90_error"

# =============================================================================
# OUTPUTS
# =============================================================================
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
    print(f"LR Scheduler         : {LR_SCHEDULER}")
    print(f"LR Min               : {LR_MIN}")
    print(f"CAP States           : {CAP_MAPPING}")
    print(f"LTC Hidden           : {LTC_HIDDEN}")
    print(f"Encoder Hidden       : {ENCODER_HIDDEN}")
    print(f"DT                   : {DT}")
    print(f"TAU Base Init        : {TAU_BASE_INIT}  (was 0.5)")
    print(f"TAU CAP Init         : {TAU_CAP_INIT}")
    print(f"Latent Reg Weight    : {LATENT_REG_WEIGHT}")
    print(f"Margin Loss Weight   : {MARGIN_LOSS_WEIGHT}")
    print(f"Margin Loss Margin   : {MARGIN_LOSS_MARGIN}")
    print(f"Threshold Percentile : {THRESHOLD_PERCENTILE}")
    print(f"Subject Score Field  : {SUBJECT_SCORE_FIELD}")
    print("=" * 80 + "\n")