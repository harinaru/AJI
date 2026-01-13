from pathlib import Path
import pandas as pd

from .channel_config import CHANNELS, PHASE2_FIXED_ROOT, OUTPUT_ROOT
from .channel_extractor import extract_channel_df


# ======================================================
# SUBJECT → LABEL MAPPING (FINAL, SAFE)
# 0 = Screenable
# 1 = Rare stress-test (Ins + Brux)
# ======================================================
def subject_to_label(subject: str) -> int:
    s = subject.lower()

    # Rare stress-test regime
    if s.startswith("brux") or s.startswith("ins"):
        return 1

    # Screenable regime
    if (
        s.startswith("nfle")
        or s.startswith("plm")
        or s.startswith("sdb")
        or s.startswith("control")
        or s.startswith("c")
        or s.startswith("n")
    ):
        return 0

    # Fail-safe: never crash the pipeline
    print(f"⚠️ Unknown subject prefix '{subject}', treating as screenable")
    return 0


def build_channel_datasets():
    print("\n[Phase 4] Building channel-wise datasets (ordered + labeled)\n")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Buffer per channel
    channel_buffers = {ch: [] for ch in CHANNELS}

    for subject_dir in PHASE2_FIXED_ROOT.iterdir():
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name
        label = subject_to_label(subject)

        # Use only full-night file
        csv_files = list(subject_dir.glob("*B183*_channels_fixed.csv"))
        if not csv_files:
            print(f"⚠️ {subject}: No B183 file found, skipping")
            continue

        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)

        if "epoch_idx" not in df.columns:
            print(f"❌ {subject}: epoch_idx missing, skipping")
            continue

        print(f"\n▶ Processing subject: {subject} (label={label})")
        print(f"  Using file: {csv_path.name}")

        for ch in CHANNELS:
            try:
                ch_df = extract_channel_df(df, ch, subject)

                # 🔥 Inject label once (epoch inherits subject label)
                ch_df.insert(1, "label", label)

                channel_buffers[ch].append(ch_df)
                print(f"  ✓ {ch}: {len(ch_df)} epochs")

            except ValueError:
                print(f"  ✗ {ch}: channel not present")
                continue

    # ======================================================
    # SAVE FINAL CHANNEL CSVs (TEMPORALLY ORDERED)
    # ======================================================
    print("\n[Phase 4] Saving final channel datasets...\n")

    for ch, dfs in channel_buffers.items():
        if not dfs:
            print(f"⚠️ No data collected for channel {ch}")
            continue

        final_df = pd.concat(dfs, ignore_index=True)

        # 🔥 CRITICAL: temporal ordering per subject
        final_df = (
            final_df
            .sort_values(by=["subject", "epoch_idx"])
            .reset_index(drop=True)
        )

        out_path = OUTPUT_ROOT / f"{ch}.csv"
        final_df.to_csv(out_path, index=False)

        print(f"✅ Saved {ch}.csv → {len(final_df)} rows (ordered + labeled)")

    print("\n🎉 Phase 4 completed successfully.\n")


if __name__ == "__main__":
    build_channel_datasets()
