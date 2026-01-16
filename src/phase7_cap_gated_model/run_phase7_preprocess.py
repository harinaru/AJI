from pathlib import Path
import numpy as np
from .phase7_feature_preprocess import preprocess_channel_csv
from .config import Config
config = Config()

for channel in config.CHANNELS:
    print(f"\n🔧 Preprocessing channel: {channel}")

    features = list(
        np.loadtxt(
            config.PHASE5_FEATURE_DIR / f"{channel}_selected_features.csv",
            dtype=str,
            delimiter=",",
            skiprows=1,
        )
    )

    preprocess_channel_csv(
        input_csv=config.PHASE4_DATA_DIR / f"{channel}.csv",
        output_csv=config.PHASE7_DATA_DIR / f"{channel}_clean.csv",
        feature_cols=features,
        zero_is_meaningful=[
            "cap_a_ratio",
            "cap_sleep_stage_counts_NREM",
            "cap_sleep_stage_counts_REM",
            "cap_sleep_stage_counts_Wake",
        ],
    )
