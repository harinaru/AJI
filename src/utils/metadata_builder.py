# src/utils/metadata_builder.py
import os
import json
from itertools import combinations
from typing import Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def save_subject_metadata(subject_id: str, metadata: Dict[str, Any],
                          base_dir: str = "results/phase1_preprocessing/json_metadata") -> str:
    """
    Save per-subject metadata as a JSON file and return its path.
    """
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f"{subject_id}_metadata.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"Saved metadata for {subject_id} → {out_path}")
    return out_path


def build_unified_csv(all_metadata: List[Dict[str, Any]],
                      out_csv: str = "results/phase1_preprocessing/tables/unified_subject_summary.csv") -> pd.DataFrame:
    """
    Combine a list of per-subject metadata dicts into a single CSV and return the DataFrame.
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not all_metadata:
        logger.warning("No metadata provided to build_unified_csv() — creating empty CSV.")
        pd.DataFrame().to_csv(out_csv, index=False)
        return pd.DataFrame()

    # create DataFrame and write
    df = pd.DataFrame(all_metadata)
    df.to_csv(out_csv, index=False)
    logger.success(f"Unified metadata CSV created → {out_csv}")
    return df

def analyze_channels(all_metadata: List[Dict[str, Any]],
                     base_dir: str = "results/phase1_preprocessing/tables") -> Dict[str, Any]:
    """
    Analyze Phase-1 metadata: channel overlap, sampling rate, duration, and more.
    Creates CSV and visual summaries for easy interpretation.

    OUTPUT FILES:
    - common_channels_all.csv
    - pairwise_common_channels.csv
    - channel_occurrence_matrix.csv
    - channel_occurrence_heatmap.png
    - sampling_duration_summary.csv
    - unified_metadata_complete.csv
    """
    os.makedirs(base_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Build subject metadata dictionaries
    # -------------------------------------------------------------------------
    channel_sets: Dict[str, set] = {}
    srates: Dict[str, float] = {}
    durations_hr: Dict[str, float] = {}
    total_samples: Dict[str, int] = {}
    extensions: Dict[str, str] = {}

    for m in all_metadata:
        subj = str(m.get("subject") or m.get("subject_id") or "unknown")
        chs_raw = m.get("ch_names") or m.get("channel_names") or []
        chs = set(chs_raw) if isinstance(chs_raw, (list, tuple, set)) else set(str(chs_raw).split(","))
        channel_sets[subj] = chs

        sf = float(m.get("sfreq", 0))
        srates[subj] = sf
        n_samples = int(m.get("n_samples", 0))
        total_samples[subj] = n_samples
        durations_hr[subj] = round((n_samples / sf / 3600.0), 3) if sf > 0 else 0.0

        src = str(m.get("source_path", ""))
        ext = os.path.splitext(src)[-1].lower().lstrip(".") or "unknown"
        extensions[subj] = ext

    if not channel_sets:
        logger.warning("⚠️ No valid metadata available — skipping analysis.")
        return {}

    subjects = list(channel_sets.keys())

    # -------------------------------------------------------------------------
    # 1️⃣ Common channels across ALL subjects
    # -------------------------------------------------------------------------
    if len(channel_sets) > 1:
        common_all = sorted(set.intersection(*channel_sets.values()))
    else:
        common_all = sorted(next(iter(channel_sets.values())))

    common_all_path = os.path.join(base_dir, "common_channels_all.csv")
    pd.Series(common_all, name="common_channel").to_csv(common_all_path, index=False)
    logger.info(f"📊 Common channels across all subjects → {len(common_all)} found")

    # -------------------------------------------------------------------------
    # 2️⃣ Pairwise common channels
    # -------------------------------------------------------------------------
    pairwise_csv = os.path.join(base_dir, "pairwise_common_channels.csv")
    pairwise_records: List[Dict[str, Any]] = []

    for (s1, ch1), (s2, ch2) in combinations(channel_sets.items(), 2):
        common = sorted(ch1 & ch2)
        pairwise_records.append({
            "subject_a": s1,
            "subject_b": s2,
            "count_common": len(common),
            "common_channels": ", ".join(common),
            "unique_to_a": ", ".join(sorted(ch1 - ch2)),
            "unique_to_b": ", ".join(sorted(ch2 - ch1))
        })

    if pairwise_records:
        pd.DataFrame(pairwise_records).to_csv(pairwise_csv, index=False)
        logger.info(f"📈 Pairwise channel analysis written for {len(pairwise_records)} pairs → {pairwise_csv}")
    else:
        # Create an empty file — this is where your IDE shows the “list[str]” issue
        empty_cols = ["subject_a", "subject_b", "count_common", "common_channels", "unique_to_a", "unique_to_b"]
        empty_df = pd.DataFrame(columns=pd.Index(empty_cols, dtype=object))  # ✅ fixed for Pyright
        empty_df.to_csv(pairwise_csv, index=False)
        logger.info(f"ℹ️ No pairwise comparison possible (only 1 subject). Empty CSV created → {pairwise_csv}")

    # -------------------------------------------------------------------------
    # 3️⃣ Channel occurrence matrix (subjects × channels)
    # -------------------------------------------------------------------------
    all_channels = sorted(set().union(*channel_sets.values()))
    idx = pd.Index(subjects, name="subject", dtype=object)
    cols = pd.Index(all_channels, dtype=object)

    matrix = pd.DataFrame(0, index=idx, columns=cols)
    for subj, chs in channel_sets.items():
        present = [c for c in chs if c in cols]
        matrix.loc[subj, present] = 1

    matrix_csv = os.path.join(base_dir, "channel_occurrence_matrix.csv")
    matrix.to_csv(matrix_csv)
    logger.info(f"🧮 Channel occurrence matrix saved → {matrix_csv}")

    # -------------------------------------------------------------------------
    # 4️⃣ Heatmap visualization
    # -------------------------------------------------------------------------
    heatmap_path = os.path.join(base_dir, "channel_occurrence_heatmap.png")
    try:
        if len(all_channels) == 0:
            raise ValueError("No channels available for plotting.")
        plt.figure(figsize=(max(8, len(all_channels) * 0.3), max(4, len(subjects) * 0.4)))
        sns.heatmap(matrix, cmap="YlGnBu", cbar=True, linewidths=0.4)
        plt.title("Channel Occurrence Heatmap", fontsize=13)
        plt.xlabel("Channels")
        plt.ylabel("Subjects")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=200)
        plt.close()
        logger.info(f"🖼️ Channel occurrence heatmap saved → {heatmap_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not generate heatmap: {e}")
        heatmap_path = "N/A"

    # -------------------------------------------------------------------------
    # 5️⃣ Sampling rate and duration summary
    # -------------------------------------------------------------------------
    sampling_summary = pd.DataFrame({
        "subject": subjects,
        "sampling_rate_Hz": [srates[s] for s in subjects],
        "duration_hr": [durations_hr[s] for s in subjects],
        "total_samples": [total_samples[s] for s in subjects],
        "file_type": [extensions[s] for s in subjects]
    })

    sampling_csv = os.path.join(base_dir, "sampling_duration_summary.csv")
    sampling_summary.to_csv(sampling_csv, index=False)
    logger.info(f"⏱️ Sampling and duration summary saved → {sampling_csv}")

    # -------------------------------------------------------------------------
    # 6️⃣ Merge everything — unified metadata summary
    # -------------------------------------------------------------------------
    unified_df = sampling_summary.merge(matrix.reset_index(), on="subject", how="left")
    unified_csv = os.path.join(base_dir, "unified_metadata_complete.csv")
    unified_df.to_csv(unified_csv, index=False)
    logger.success(f"✅ Unified metadata summary saved → {unified_csv}")

    # -------------------------------------------------------------------------
    # 7️⃣ Return summary paths
    # -------------------------------------------------------------------------
    return {
        "common_all": common_all,
        "n_subjects": len(subjects),
        "pairwise_csv": pairwise_csv,
        "matrix_csv": matrix_csv,
        "heatmap_png": heatmap_path,
        "sampling_csv": sampling_csv,
        "unified_csv": unified_csv,
    }