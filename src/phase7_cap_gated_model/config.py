"""
PHASE 7 — CONFIGURATION FILE
CAP-Gated Temporal Normality Model

This file:
✔ Defines all hyperparameters
✔ Checks GPU availability
✔ Prints full experimental logic
✔ Creates required directories
✔ Ensures reproducibility

This file DOES NOT train any model.
"""

from pathlib import Path
import torch
import numpy as np
import random
from datetime import datetime


class Config:
    # ==========================================================
    # 1. EXPERIMENT IDENTITY
    # ==========================================================
    EXPERIMENT_NAME = "Phase7_CAP_Gated_Temporal_Model"
    VERSION = "v1.0"
    DATE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ==========================================================
    # 2. PATHS (INPUT & OUTPUT)
    # ==========================================================
    ROOT_DIR = Path("results")

    PHASE4_DATA_DIR = ROOT_DIR / "phase4_channel_datasets"
    PHASE5_FEATURE_DIR = ROOT_DIR / "phase5_feature_selection"
    PHASE7_DATA_DIR = ROOT_DIR / "phase7_cap_gated_model" / "data"

    OUTPUT_DIR = ROOT_DIR / "phase7_cap_gated_model"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    FIGURE_DIR = OUTPUT_DIR / "figures"

    # ==========================================================
    # 3. DATA SETTINGS
    # ==========================================================
    CHANNELS = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]

    SUBJECT_COL = "subject"
    LABEL_COL = "label"
    CAP_COL = "cap_present"
    EPOCH_COL = "epoch_idx"

    N_FEATURES = 45          # Selected in Phase 5
    INPUT_DIM = 45           # Per-channel
    OUTPUT_DIM = 45          # Next-step reconstruction

    # ==========================================================
    # 4. MODEL ARCHITECTURE
    # ==========================================================
    HIDDEN_DIM = 64           # Temporal state size

    # CAP-Gated Memory (Physiology-informed)
    ALPHA_A_INIT = 0.9        # Retention during CAP (instability)
    ALPHA_B_INIT = 0.5        # Retention during non-CAP (stability)

    # ==========================================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================================
    BATCH_SIZE = 16           # Subjects per batch
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0

    GRAD_CLIP = 1.0
    EARLY_STOPPING_PATIENCE = 5

    # ==========================================================
    # 6. DATA SPLIT (SCARCITY-SAFE)
    # ==========================================================
    TRAIN_RATIO = 0.8         # Controls only
    VAL_RATIO = 0.2           # Controls only
    RARE_USED_ONLY_FOR_TEST = True

    # ==========================================================
    # 7. RANDOM SEED (REPRODUCIBILITY)
    # ==========================================================
    SEED = 42

    # ==========================================================
    # 8. DEVICE CONFIGURATION
    # ==========================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================
    # 9. INITIALIZATION METHODS
    # ==========================================================
    @classmethod
    def set_seed(cls):
        """Ensure full reproducibility"""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def create_directories(cls):
        """Create output directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # 10. TERMINAL LOGIC PRINT (VERY IMPORTANT)
    # ==========================================================
    @classmethod
    def print_experiment_summary(cls):
        print("\n" + "=" * 80)
        print(" PHASE 7 — CAP-GATED TEMPORAL NORMALITY MODEL ")
        print("=" * 80)

        print(f"\n🧪 Experiment      : {cls.EXPERIMENT_NAME}")
        print(f"🔖 Version         : {cls.VERSION}")
        print(f"📅 Run Timestamp   : {cls.DATE}")

        print("\n🔍 PROBLEM FORMULATION")
        print("-" * 40)
        print("• Task            : Temporal normality learning (self-supervised)")
        print("• Prediction      : Next-step EEG feature reconstruction")
        print("• Labels used     : ONLY for evaluation (NOT training)")
        print("• Rare disorders  : Treated as out-of-distribution (OOD)")
        print("• CAP usage       : Controls temporal memory (CORE NOVELTY)")

        print("\n🧠 MODEL DESIGN")
        print("-" * 40)
        print(f"• Input features  : {cls.INPUT_DIM}")
        print(f"• Hidden state    : {cls.HIDDEN_DIM}")
        print(f"• Output          : {cls.OUTPUT_DIM}")
        print(f"• CAP α_A (init)  : {cls.ALPHA_A_INIT}  (retain during CAP)")
        print(f"• CAP α_B (init)  : {cls.ALPHA_B_INIT}  (forget during non-CAP)")

        print("\n📊 DATA & SPLIT")
        print("-" * 40)
        print(f"• Channels        : {', '.join(cls.CHANNELS)}")
        print(f"• Train split     : {int(cls.TRAIN_RATIO * 100)}% controls")
        print(f"• Val split       : {int(cls.VAL_RATIO * 100)}% controls")
        print(f"• Rare disorders  : NEVER used in training")

        print("\n⚙️ TRAINING SETTINGS")
        print("-" * 40)
        print(f"• Batch size      : {cls.BATCH_SIZE} subjects")
        print(f"• Epochs (max)    : {cls.NUM_EPOCHS}")
        print(f"• Learning rate   : {cls.LEARNING_RATE}")
        print(f"• Early stopping  : {cls.EARLY_STOPPING_PATIENCE} epochs")
        print(f"• Gradient clip   : {cls.GRAD_CLIP}")

        print("\n🖥️ DEVICE STATUS")
        print("-" * 40)
        print(f"• CUDA available  : {torch.cuda.is_available()}")
        print(f"• Selected device : {cls.DEVICE}")
        if torch.cuda.is_available():
            print(f"• GPU name        : {torch.cuda.get_device_name(0)}")

        print("\n📁 OUTPUT LOCATIONS")
        print("-" * 40)
        print(f"• Checkpoints     : {cls.CHECKPOINT_DIR}")
        print(f"• Logs            : {cls.LOG_DIR}")
        print(f"• Figures         : {cls.FIGURE_DIR}")

        print("\n✅ CONFIGURATION LOADED SUCCESSFULLY")
        print("=" * 80 + "\n")
