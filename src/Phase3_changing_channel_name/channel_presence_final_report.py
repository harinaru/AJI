from pathlib import Path
import pandas as pd
import json

TARGET_CHANNELS = [
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]


def select_b183_csv(subject_dir: Path):
    for csv in subject_dir.glob("*_channels_fixed.csv"):
        if "B183" in csv.name:
            return csv
    return None


def generate_final_channel_report(mapped_root: Path, out_json: Path):
    report = {
        "summary": {
            "total_subjects": 0,
            "subjects_with_all_4_channels": 0,
            "subjects_missing_any_channel": 0
        },
        "subjects": {}
    }

    subject_dirs = [d for d in mapped_root.iterdir() if d.is_dir()]
    report["summary"]["total_subjects"] = len(subject_dirs)

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        csv_file = select_b183_csv(subject_dir)

        if csv_file is None:
            report["subjects"][subject] = {
                "status": "SKIPPED",
                "reason": "B183 CSV not found"
            }
            report["summary"]["subjects_missing_any_channel"] += 1
            continue

        df = pd.read_csv(csv_file, nrows=1)
        columns = df.columns.tolist()

        channel_info = {}
        has_all = True

        for ch in TARGET_CHANNELS:
            ch_cols = [c for c in columns if c.startswith(ch + "_")]
            channel_info[ch] = {
                "present": len(ch_cols) > 0,
                "feature_count": len(ch_cols)
            }
            if not ch_cols:
                has_all = False

        report["subjects"][subject] = {
            "csv_used": csv_file.name,
            "channels": channel_info,
            "has_all_4": has_all
        }

        if has_all:
            report["summary"]["subjects_with_all_4_channels"] += 1
        else:
            report["summary"]["subjects_missing_any_channel"] += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Final channel presence report saved to:\n{out_json}\n")


if __name__ == "__main__":
    mapped_root = Path("results/phase2_features_fixed")
    out_json = Path("results/phase2_features_fixed/final_channel_presence_report.json")

    generate_final_channel_report(mapped_root, out_json)
