import json
from pathlib import Path
import mne

def extract_channels_from_edf(edf_path: Path) -> dict:
    """
    Read EDF header only and extract channel names in correct order.
    """
    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose=False
    )

    return {
        "subject": edf_path.stem,
        "edf_file": edf_path.name,
        "n_channels": len(raw.ch_names),
        "channel_names": raw.ch_names
    }


def run_edf_scan(raw_dir: Path, out_dir: Path):
    """
    Scan raw EDF directory and generate channel reference JSONs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(
        [f for f in raw_dir.iterdir() if f.suffix in (".edf", ".EDF")]
    )

    print(f"\n🔍 EDF files found: {len(edf_files)}\n")

    if not edf_files:
        print("❌ No EDF files found. Check RAW_EDF_DIR.")
        return

    for edf_path in edf_files:
        try:
            info = extract_channels_from_edf(edf_path)

            out_file = out_dir / f"{info['subject']}_channels.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)

            print(
                f"✅ {info['subject']}: "
                f"{info['n_channels']} channels saved"
            )

        except Exception as e:
            print(f"❌ Failed for {edf_path.name}: {e}")
