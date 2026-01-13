import json
import pandas as pd
from pathlib import Path

TARGET_CHANNELS = [
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]


def convert_channel_report_to_csv(report_json: Path, out_csv: Path):
    with open(report_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    rows = []

    for subject, info in report["subjects"].items():
        if info.get("status") == "SKIPPED":
            continue

        row = {
            "subject": subject,
            "csv_used": info.get("csv_used", "")
        }

        for ch in TARGET_CHANNELS:
            ch_info = info["channels"].get(ch, {})
            row[f"{ch}_present"] = int(ch_info.get("present", False))
            row[f"{ch}_feature_count"] = ch_info.get("feature_count", 0)

        row["has_all_4_channels"] = int(info.get("has_all_4", False))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values("subject", inplace=True)

    df.to_csv(out_csv, index=False)
    print(f"\n✅ Channel presence table saved to:\n{out_csv}\n")


if __name__ == "__main__":
    report_json = Path("results/phase2_features_fixed/final_channel_presence_report.json")
    out_csv = Path("results/phase2_features_fixed/final_channel_presence_table.csv")

    convert_channel_report_to_csv(report_json, out_csv)
