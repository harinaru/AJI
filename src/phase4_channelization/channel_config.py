from pathlib import Path

# Input: mapped CSVs from Phase 3
PHASE2_FIXED_ROOT = Path("results/phase2_features_fixed")

# Output: channel-wise datasets
OUTPUT_ROOT = Path("results/phase4_channel_datasets")

CHANNELS = [
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]

# Columns that must always be kept (shared)
COMMON_COLUMNS = [
    "subject",
    "epoch_idx",
    "epoch_start_s",
    "epoch_end_s",
    "epoch_overlap_s",
    "epoch_duration_s",
    "has_overlap",

    "global_mean",
    "global_std",
    "global_energy",
    "global_rms",

    "cap_present",
    "cap_subtype",
    "cap_a_ratio",
    "cap_A_start",
    "cap_A_duration",
    "cap_A1_time",
    "cap_A2_time",
    "cap_A3_time",
    "cap_A1_ratio",
    "cap_A2_ratio",
    "cap_A3_ratio",
    "cap_CAPtime",
    "cap_CAPrate",
    "cap_CAPdensity",
    "cap_CAPsequences",
    "cap_NREMtime",
    "cap_A1_sequences",
    "cap_A2_sequences",
    "cap_A3_sequences",
    "cap_sleep_stage_counts_NREM",
    "cap_sleep_stage_counts_REM",
    "cap_sleep_stage_counts_Wake",
]
