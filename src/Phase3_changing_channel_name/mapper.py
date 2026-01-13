import json
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm   # ✅ ADD THIS


def extract_ch_indices(columns):
    pattern = re.compile(r"ch(\d+)_")
    return sorted({int(m.group(1)) for c in columns if (m := pattern.match(c))})


def validate_channels(columns, n_channels):
    found = extract_ch_indices(columns)
    expected = list(range(1, n_channels + 1))

    missing = [f"ch{i}" for i in expected if i not in found]
    extra = [f"ch{i}" for i in found if i not in expected]

    return missing, extra


def rename_columns(df, channel_names):
    new_cols = {}
    for col in df.columns:
        for i, ch_name in enumerate(channel_names, start=1):
            prefix = f"ch{i}_"
            if col.startswith(prefix):
                new_cols[col] = col.replace(prefix, f"{ch_name}_", 1)
                break
    return df.rename(columns=new_cols)


def run_mapper(
    phase2_root: Path,
    channel_ref_dir: Path,
    output_root: Path,
    report_path: Path
):
    report = {
        "summary": {
            "total_subjects": 0,
            "subjects_success": 0,
            "subjects_failed": 0
        },
        "subjects": {}
    }

    subjects = [d for d in phase2_root.iterdir() if d.is_dir()]
    report["summary"]["total_subjects"] = len(subjects)

    # ✅ SUBJECT-LEVEL PROGRESS BAR
    for subj_dir in tqdm(subjects, desc="🧠 Mapping subjects"):
        subject = subj_dir.name
        subj_report = {}

        channel_json = channel_ref_dir / f"{subject}_channels.json"
        if not channel_json.exists():
            subj_report.update({
                "status": "FAILURE",
                "message": "No channel reference JSON found"
            })
            report["summary"]["subjects_failed"] += 1
            report["subjects"][subject] = subj_report
            continue

        with open(channel_json, "r") as f:
            ch_ref = json.load(f)

        n_channels = ch_ref["n_channels"]
        channel_names = ch_ref["channel_names"]

        csv_files = [f for f in subj_dir.glob("*.csv")]
        if not csv_files:
            subj_report.update({
                "n_channels": n_channels,
                "status": "FAILURE",
                "message": "No valid CSV files found"
            })
            report["summary"]["subjects_failed"] += 1
            report["subjects"][subject] = subj_report
            continue

        subj_details = {}
        success_files = []

        # ✅ CSV-LEVEL PROGRESS BAR (nested)
        for csv_path in tqdm(csv_files, desc=f"📄 {subject} CSVs", leave=False):
            df = pd.read_csv(csv_path)
            missing, extra = validate_channels(df.columns, n_channels)

            if missing or extra:
                subj_details[csv_path.name] = {
                    "missing_channels": missing,
                    "extra_channels": extra
                }
                continue

            fixed_df = rename_columns(df, channel_names)

            out_dir = output_root / subject
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / csv_path.name.replace(".csv", "_channels_fixed.csv")
            fixed_df.to_csv(out_file, index=False)

            success_files.append(csv_path.name)

        if success_files:
            subj_report.update({
                "n_channels": n_channels,
                "status": "SUCCESS",
                "message": f"{subject}: {n_channels} channels updated successfully",
                "processed_csvs": success_files
            })
            report["summary"]["subjects_success"] += 1
        else:
            subj_report.update({
                "n_channels": n_channels,
                "status": "FAILURE",
                "message": "Channel mismatch detected",
                "details": subj_details
            })
            report["summary"]["subjects_failed"] += 1

        report["subjects"][subject] = subj_report

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Phase-3 mapping report saved to: {report_path}\n")
