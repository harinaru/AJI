from .config import RAW_EDF_DIR, CHANNEL_REF_DIR
from .edf_reader import run_edf_scan
from .mapper import run_mapper
from pathlib import Path


def main():
    print("\n==============================")
    print(" Phase-3: EDF Channel Extract ")
    print("==============================\n")

    print(f"RAW EDF directory : {RAW_EDF_DIR.resolve()}")
    print(f"Output JSON dir   : {CHANNEL_REF_DIR.resolve()}\n")

    run_edf_scan(RAW_EDF_DIR, CHANNEL_REF_DIR)

    print("\n✅ Phase-3 EDF extraction completed.\n")
    run_mapper(
    phase2_root=Path("results/phase2_features"),
    channel_ref_dir=Path("src/Phase3_changing_channel_name/channel_refs"),
    output_root=Path("results/phase2_features_fixed"),
    report_path=Path("results/phase2_features_fixed/channel_mapping_report.json")
)



if __name__ == "__main__":
    main()
