# ===============================================================
# src/core/pipeline_controller.py
# v4.4.9_research_final — Pyright-clean unified EEG + CAP + Terzano + Phase2
# ===============================================================
import os
import json
import traceback
from typing import List, Optional, Dict, Any, Tuple, Union, Sequence, cast
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# ===============================================================
# Core utilities
# ===============================================================
from utils.folder_creation import verify_all
from utils.logger import get_logger
from utils.progress_tracker import status

# ===============================================================
# Phase 1 — Preprocessing
# ===============================================================
from phase1_preprocessing.data_loader import discover_subjects, load_subject_normalized
from phase1_preprocessing.filtering import apply_filters, detect_artifacts_threshold
from phase1_preprocessing.epoching import make_epochs, label_epochs, save_processed_subject
from phase1_preprocessing.st_annotation_parser import parse_st_file, map_caps_to_epochs
from phase1_preprocessing.scoring_reader import read_remlogic_txt
from phase1_preprocessing.cap_metrics import compute_terzano_metrics

# ===============================================================
# Phase 2 — Feature Extraction
# ===============================================================
from phase2_features.feature_manager import extract_features_for_cohort

# ===============================================================
# Reproducibility
# ===============================================================
SEED = 42
np.random.seed(SEED)

# ===============================================================
# Type definitions (global to avoid Pyright internal errors)
# ===============================================================
ResultType = Union[
    Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]],
    Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float], Dict[str, int]]
]


# =====================================================================
# 🧩 Phase 1 — EEG Preprocessing + CAP & Hypnogram Integration
# =====================================================================
def run_phase1_from_main(
    cfg: Dict[str, Any],
    raw_dir: str,
    out_dir: str,
    subjects: Optional[List[str]] = None,
    workers: int = 1,
    debug: bool = False
) -> None:
    """Complete Phase-1 pipeline integrating CAP and hypnogram."""
    if debug:
        logger.info("⚙️ Debug mode enabled")

    verify_all()
    raw_dir, out_dir = os.path.abspath(raw_dir), os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_subjects = discover_subjects(raw_dir)
    subject_list = [s for s in all_subjects if s in (subjects or all_subjects)]
    if not subject_list:
        logger.warning("⚠️ No subjects found — check raw directory.")
        return

    summary_records: List[Dict[str, Any]] = []
    status(f"Processing {len(subject_list)} subjects...")

    for subid in tqdm(subject_list, desc="Processing subjects..."):
        if subid.endswith(".st"):
            continue

        record: Dict[str, Any] = {"subject": subid}
        try:
            # ==================================================
            # Load + filter data
            # ==================================================
            subj = load_subject_normalized(subid, raw_dir)
            data, sfreq = subj["data"], float(subj["sfreq"])
            ch_names = subj.get("ch_names", [])
            annotations = subj.get("annotations", [])

            filtered, filt_meta = apply_filters(
                data, sfreq,
                low=float(cfg.get("filter_low", 0.3)),
                high=float(cfg.get("filter_high", 35.0)),
                notch_freqs=cfg.get("notch_freqs", [50.0])
            )

            bad_mask = detect_artifacts_threshold(
                filtered, sfreq,
                thresh_uV=float(cfg.get("artifact_thresh_uV", 200.0))
            )

            # ==================================================
            # Epoching + labeling
            # ==================================================
            epoch_len = int(float(cfg.get("epoch_length_s", 30)))
            step_s = int(float(cfg.get("epoch_step_s", 5)))

            epochs, epoch_times = make_epochs(
                filtered, sfreq,
                epoch_length_s=epoch_len,
                step_s=step_s,
                bad_mask=bad_mask
            )

            epoch_labels = label_epochs(
                epoch_times, annotations,
                horizon_s=int(float(cfg.get("prediction_horizon_s", 30)))
            )

            subject_out = os.path.join(out_dir, subid)
            os.makedirs(subject_out, exist_ok=True)

            save_processed_subject(
                subject_out, subid, filtered, sfreq,
                ch_names, epochs, epoch_times, epoch_labels, annotations
            )

            # --------------------------------------------------
            # Metadata initialization
            # --------------------------------------------------
            meta: Dict[str, Any] = {
                "subject": subid,
                "sfreq": sfreq,
                "n_channels": int(filtered.shape[0]),
                "n_samples": int(filtered.shape[1]),
                "filter_meta": filt_meta,
                "epoch_length_s": epoch_len,
                "epoch_step_s": step_s,
                "annotation_sources": []
            }

            # ==================================================
            # CAP Annotation (.st)
            # ==================================================
            annotation_path = os.path.join(raw_dir, f"{subid}.edf.st")
            if os.path.exists(annotation_path):
                try:
                    cap_json_path = os.path.join(subject_out, f"{subid}_cap_annotations.json")
                    cap_annotations = parse_st_file(annotation_path, output_json=cap_json_path)
                    epoch_times_loaded = np.load(os.path.join(subject_out, "epoch_times.npy"), allow_pickle=True)
                    epoch_labels_cap = map_caps_to_epochs(
                        cap_annotations, epoch_times_loaded,
                        epoch_step_s=step_s, n_epochs=len(epoch_times_loaded)
                    )
                    np.save(os.path.join(subject_out, "epoch_labels.npy"), epoch_labels_cap)
                    meta["annotation_sources"].append(".st")
                except Exception as e:
                    logger.warning(f"❌ .st CAP parsing failed for {subid}: {e}")

            # ==================================================
            # REMlogic (.txt) + Terzano metrics
            # ==================================================
            txt_path = os.path.join(raw_dir, f"{subid}.txt")
            if os.path.exists(txt_path):
                try:
                    result = read_remlogic_txt(
                        txt_path,
                        output_json=os.path.join(subject_out, f"{subid}_cap_annotations.json")
                    )

                    # Handle both 4- and 5-return variants
                    hyp, time_tot, duration, type_ar, sleep_stage_counts = [], [], [], [], {"NREM": 0, "REM": 0, "Wake": 0}
                    if isinstance(result, tuple):
                        if len(result) == 5:
                            hyp, time_tot, duration, type_ar, sleep_stage_counts = result  # type: ignore
                        elif len(result) == 4:
                            hyp, time_tot, duration, type_ar = result  # type: ignore
                        else:
                            logger.warning("Unexpected tuple size from read_remlogic_txt(); skipping Terzano.")

                    # Normalize
                    time_tot_arr = np.array(time_tot, dtype=float) if isinstance(time_tot, (list, tuple)) else np.asarray(time_tot, dtype=float)
                    duration_arr = np.array(duration, dtype=float) if isinstance(duration, (list, tuple)) else np.asarray(duration, dtype=float) # type: ignore # noqa: F821
                    hyp_arr = np.array(hyp, dtype=int) if isinstance(hyp, (list, tuple)) else np.asarray(hyp, dtype=int) # type: ignore # noqa: F821    
                    # Compute Terzano metrics safely
                    if time_tot_arr.size > 0 and duration_arr.size > 0:
                        cap_metrics = compute_terzano_metrics(
                            time_tot_arr.tolist(),
                            duration_arr.tolist(),
                            hyp_arr.tolist()
                        )
                    else:
                        cap_metrics = {"CAPtime": 0.0, "CAPrate": 0.0, "CAPdensity": 0.0}
                    
                    # Add stage counts (nested dict, Pyright-safe)
                    cap_metrics["sleep_stage_counts"] = {  # type: ignore[reportArgumentType]
                        "NREM": int(sleep_stage_counts.get("NREM", 0)),
                        "REM": int(sleep_stage_counts.get("REM", 0)),
                        "Wake": int(sleep_stage_counts.get("Wake", 0))
                    }
                    # Save CAP metrics
                    metrics_path = os.path.join(subject_out, f"{subid}_cap_metrics.json")
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(cap_metrics, f, indent=2)

                    meta["annotation_sources"].append(".txt")
                    meta["cap_metrics_file"] = os.path.basename(metrics_path)
                    logger.success(f"✅ CAP metrics saved for {subid}")

                except Exception as e:
                    logger.warning(f"❌ REMlogic parsing failed for {subid}: {e}")

            # ==================================================
            # Save metadata
            # ==================================================
            meta_path = os.path.join(subject_out, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # Record subject summary
            record.update({
                "status": "OK",
                "n_epochs": int(epochs.shape[0]),
                "pct_bad_segments": float(np.mean(bad_mask)),
                "message": ""
            })

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"❌ Error processing {subid}: {e}\n{tb}")
            record.update({
                "status": "FAILED",
                "n_epochs": 0,
                "pct_bad_segments": 0.0,
                "message": str(e)
            })
        finally:
            summary_records.append(record)

    # ==================================================
    # Save processing summary
    # ==================================================
    summary_df = pd.DataFrame(summary_records)
    out_table_dir = os.path.join("results", "phase1_preprocessing", "tables")
    os.makedirs(out_table_dir, exist_ok=True)
    summary_path = os.path.join(out_table_dir, "processing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.success(f"📜 Phase 1 summary saved → {summary_path}")
    status("Phase 1 completed successfully.")


# =====================================================================
# 🧠 Phase 2 — Feature Extraction
# =====================================================================
def run_phase2_from_main(cfg: dict, processed_dir: str, results_dir: str):
    """Phase 2: Extract per-epoch features."""
    processed_dir, results_dir = os.path.abspath(processed_dir), os.path.abspath(results_dir)
    if not os.path.exists(processed_dir):
        logger.error(f"Processed directory missing: {processed_dir}")
        return
    extract_features_for_cohort(processed_dir, results_dir)
    logger.success("✅ Phase 2: Feature extraction completed successfully.")
