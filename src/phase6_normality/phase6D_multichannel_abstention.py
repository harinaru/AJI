from pathlib import Path
import pandas as pd
import numpy as np

# ================= CONFIG ================= #

CHANNELS = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]

TAIL_DIR = Path("results/phase6B_tail")
OUT_DIR = Path("results/phase6D_abstention")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_COL = "subject"
LABEL_COL = "label"

# ========================================== #


def load_tail_uncertainty(channel):
    path = TAIL_DIR / f"tail_uncertainty_{channel}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase-6B file: {path}")
    return pd.read_csv(path)


def main():
    print("\n==============================")
    print(" PHASE 6D — MULTI-CHANNEL ABSTENTION")
    print("==============================\n")

    channel_dfs = []

    # ---------- Load each channel ----------
    for ch in CHANNELS:
        print(f"→ Loading {ch}")
        df = load_tail_uncertainty(ch)

        # Keep only what we need
        df = df[[SUBJECT_COL, LABEL_COL, "uncertainty_p90"]].copy()
        df.rename(
            columns={"uncertainty_p90": f"p90_{ch}"},
            inplace=True
        )

        channel_dfs.append(df)

    # ---------- Merge across channels ----------
    merged = channel_dfs[0]
    for df in channel_dfs[1:]:
        merged = merged.merge(
            df,
            on=[SUBJECT_COL, LABEL_COL],
            how="inner"
        )

    print(f"\n✓ Merged subjects: {len(merged)}")

    # ---------- Channel-specific thresholds ----------
    thresholds = {}
    for ch in CHANNELS:
        ctrl_vals = merged[merged[LABEL_COL] == 0][f"p90_{ch}"].values
        thresholds[ch] = np.percentile(ctrl_vals, 95)

    print("\nChannel-wise τ (95th percentile):")
    for ch, t in thresholds.items():
        print(f"  τ_{ch} = {t:.3f}")

    # ---------- OR-based abstention ----------
    def abstain(row):
        for ch in CHANNELS:
            if row[f"p90_{ch}"] > thresholds[ch]:
                return 1
        return 0

    merged["abstain"] = merged.apply(abstain, axis=1)

    # ---------- Summary ----------
    summary = (
        merged.groupby(LABEL_COL)["abstain"]
        .agg(["mean", "count"])
        .reset_index()
    )

    print("\nAbstention Summary:")
    print(summary)

    # ---------- Save ----------
    merged.to_csv(OUT_DIR / "phase6D_subject_abstention.csv", index=False)
    summary.to_csv(OUT_DIR / "phase6D_abstention_summary.csv", index=False)

    print("\n✅ Phase-6D completed successfully\n")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
