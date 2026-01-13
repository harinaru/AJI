from pathlib import Path
import pandas as pd
import json

# ---------------- CONFIG ---------------- #

CHANNELS = ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]

PHASE6_DIR = Path("results/phase6_normality")
OUT_DIR = PHASE6_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
SUBJECT_COL = "subject"

# --------------------------------------- #

def load_tail_uncertainty(channel):
    """
    Load Phase-6B tail uncertainty file (robust to naming)
    """
    candidates = [
        PHASE6_DIR / f"tail_uncertainty_{channel}.csv",
        PHASE6_DIR / f"phase6B_tail_uncertainty_{channel}.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"  ✓ Loaded {path.name}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"Missing tail uncertainty file for {channel}. Tried: {candidates}"
    )


def main():
    print("\n==============================")
    print(" PHASE 6D — MULTI-CHANNEL ABSTENTION ")
    print("==============================\n")

    # Load thresholds
    thresholds = json.load(open(PHASE6_DIR / "thresholds.json"))

    dfs = []

    # Load all channel-wise uncertainty tables
    for ch in CHANNELS:
        print(f"→ Loading {ch}")
        df = load_tail_uncertainty(ch)

        # Standardize column names
        df = df.rename(columns={
            "uncertainty_p90": f"u_{ch}",
            "uncertainty": f"u_{ch}"
        })

        df = df[[SUBJECT_COL, LABEL_COL, f"u_{ch}"]]
        dfs.append(df)

    # Merge all channels on subject
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(
            df, on=[SUBJECT_COL, LABEL_COL], how="inner"
        )

    # Abstention rule (AND over channels)
    def abstain(row):
        for ch in CHANNELS:
            if row[f"u_{ch}"] > thresholds[ch]:
                return 1
        return 0

    merged["abstain"] = merged.apply(abstain, axis=1)

    # Summary statistics
    summary = (
        merged.groupby(LABEL_COL)["abstain"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )

    summary["abstention_rate_%"] = 100 * summary["mean"]

    # Save outputs
    merged.to_csv(OUT_DIR / "phase6D_subject_abstention.csv", index=False)
    summary.to_csv(OUT_DIR / "phase6D_abstention_summary.csv", index=False)

    print("\n📊 Abstention Summary")
    print(summary)

    print("\n🎉 Phase-6D completed successfully")
    print(f"Results saved to: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
