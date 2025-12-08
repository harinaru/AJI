# src/core/pipeline_controller.py
# v4.4.9_research_final_fixed — unified EEG + CAP + Terzano + Phase2 (Pyright-clean)
import os
import json
import traceback
from typing import List, Optional, Dict, Any, Sequence, Union, Tuple, cast
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# project utilities (assume these exist in your repo)
from utils.folder_creation import verify_all
from utils.logger import get_logger
from utils.progress_tracker import status

# Phase1 preprocessing modules
from phase1_preprocessing.data_loader import discover_subjects, load_subject_normalized
from phase1_preprocessing.filtering import apply_filters, detect_artifacts_threshold
from phase1_preprocessing.epoching import make_epochs, label_epochs, save_processed_subject
from phase1_preprocessing.st_annotation_parser import parse_st_file, map_caps_to_epochs
from phase1_preprocessing.scoring_reader import read_remlogic_txt
from phase1_preprocessing.cap_metrics import compute_terzano_metrics

# Phase2
from phase2_features.feature_manager import extract_features_for_cohort

# reproducibility
SEED = 42
np.random.seed(SEED)

# Type alias used in earlier code (keeps type-checkers happy)
ResultType = Union[
    Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]],
    Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float], Dict[str, int]]
]


# ---------------------------
# Utility: ensure hyp array shape
# ---------------------------
def ensure_hyp_array(hyp: Any) -> np.ndarray:
    """
    Ensure hyp is a 2-D numpy array shape (N,2) with dtype float.
    - If hyp is None -> returns empty (0,2) array.
    - If hyp is 1-D and length is even -> reshape(-1,2).
    - If hyp is 1-D and length matches a stage count -> assume (stage,) and build start times as epoch*30s.
    - If hyp is 2-D but second dim != 2 -> attempt to fix by slice/pad.
    """
    if hyp is None:
        return np.empty((0, 2), dtype=float)

    # Use a different local name to avoid shadowing confusions for static checkers
    arr = np.asarray(hyp)

    # Already 2D with correct shape
    if getattr(arr, "ndim", 0) == 2 and arr.shape[1] == 2:
        try:
            return arr.astype(float)
        except Exception:
            return arr.astype(float, copy=True)

    # 1D cases
    if getattr(arr, "ndim", 0) == 1:
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        # even-length flattened pairs (stage, start, stage, start...)
        if arr.size % 2 == 0:
            try:
                arr2 = arr.reshape(-1, 2).astype(float)
                logger.debug(f"ensure_hyp_array: reshaped 1D hyp -> {arr2.shape}")
                return arr2
            except Exception:
                pass
        # stage-only vector: create start times every 30s
        try:
            stage_vec = arr.astype(float)
            starts = (np.arange(stage_vec.shape[0]) * 30.0).astype(float)
            arr2 = np.column_stack([stage_vec, starts])
            logger.debug(f"ensure_hyp_array: interpreted 1D hyp as stage-only -> {arr2.shape}")
            return arr2
        except Exception:
            logger.warning("ensure_hyp_array: failed to interpret 1D hyp; returning empty (0,2) array")
            return np.empty((0, 2), dtype=float)

    # 2D but wrong shape: try slice/pad
    if getattr(arr, "ndim", 0) == 2:
        if arr.shape[1] > 2:
            logger.debug("ensure_hyp_array: slicing extra columns from hyp (keeping first two).")
            return arr[:, :2].astype(float)
        elif arr.shape[1] == 1:
            col0 = arr[:, 0].astype(float)
            # If values look like absolute timestamps (large), keep as starts
            if np.nanmin(col0) >= 1000:
                st = np.zeros_like(col0)
                return np.column_stack([st, col0]).astype(float)
            else:
                starts = (np.arange(col0.shape[0]) * 30.0).astype(float)
                return np.column_stack([col0, starts]).astype(float)

    logger.warning("ensure_hyp_array: unable to coerce hyp into (N,2); returning empty array.")
    return np.empty((0, 2), dtype=float)


# =====================================================================
# Phase 1 — EEG Preprocessing + CAP & Hypnogram Integration
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
                "n_channels": int(filtered.shape[0]) if hasattr(filtered, "shape") else 0,
                "n_samples": int(filtered.shape[1]) if hasattr(filtered, "shape") else 0,
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
            if os.path.exists(txt_path) and read_remlogic_txt is not None:
                try:
                    result = read_remlogic_txt(
                        txt_path,
                        output_json=os.path.join(subject_out, f"{subid}_cap_annotations.json")
                    )

                    # normalize result tuple length to either 4 or 5 returns
                    hyp = []
                    time_tot = []
                    duration = []
                    type_ar = []
                    sleep_stage_counts = {"NREM": 0, "REM": 0, "Wake": 0}

                    if isinstance(result, tuple):
                        if len(result) == 5:
                            hyp, time_tot, duration, type_ar, sleep_stage_counts = result  # type: ignore
                        elif len(result) == 4:
                            hyp, time_tot, duration, type_ar = result  # type: ignore
                        else:
                            logger.warning("Unexpected tuple size from read_remlogic_txt(); skipping Terzano.")

                    # Coerce arrays / lists
                    time_tot_arr = np.asarray(time_tot, dtype=float) if isinstance(time_tot, (list, tuple, np.ndarray)) else np.asarray(time_tot, dtype=float)
                    duration_arr = np.asarray(duration, dtype=float) if isinstance(duration, (list, tuple, np.ndarray)) else np.asarray(duration, dtype=float)
                    type_ar_list = list(type_ar) if isinstance(type_ar, (list, tuple, np.ndarray)) else list(type_ar or [])

                    # Ensure hyp is 2-column (stage, start_seconds)
                    hyp_fixed = ensure_hyp_array(hyp)

                    logger.debug(f"[REML] hyp_fixed shape: {getattr(hyp_fixed, 'shape', None)}")

                    # compute Terzano metrics only if compute_terzano_metrics available
                    if compute_terzano_metrics is not None and time_tot_arr.size > 0 and duration_arr.size > 0:
                        # compute_terzano_metrics signature: (time_tot, duration, type_ar, hyp)
                        cap_metrics = compute_terzano_metrics(
                            time_tot_arr.tolist(),
                            duration_arr.tolist(),
                            type_ar_list,
                            hyp_fixed
                        )
                    else:
                        # safe default
                        cap_metrics = {
                            "CAPtime": 0.0, "CAPrate": 0.0, "CAPdensity": 0.0,
                            "CAPsequences": 0, "NREMtime": 0.0,
                            "A1_time": 0.0, "A2_time": 0.0, "A3_time": 0.0,
                            "A1_ratio": 0.0, "A2_ratio": 0.0, "A3_ratio": 0.0,
                            "A1_sequences": 0, "A2_sequences": 0, "A3_sequences": 0
                        }

                    # attach stage counts cleanly
                    cap_metrics["sleep_stage_counts"] = {
                        "NREM": int(sleep_stage_counts.get("NREM", 0)),
                        "REM": int(sleep_stage_counts.get("REM", 0)),
                        "Wake": int(sleep_stage_counts.get("Wake", 0))
                    }

                    # Save CAP metrics JSON
                    metrics_path = os.path.join(subject_out, f"{subid}_cap_metrics.json")
                    try:
                        with open(metrics_path, "w", encoding="utf-8") as fo:
                            json.dump(cap_metrics, fo, indent=2)
                        meta["annotation_sources"].append(".txt")
                        meta["cap_metrics_file"] = os.path.basename(metrics_path)
                        logger.success(f"✅ CAP metrics saved for {subid}")
                    except Exception as e:
                        logger.warning(f"Could not save CAP metrics JSON for {subid}: {e}")

                except Exception as e:
                    logger.warning(f"❌ REMlogic parsing failed for {subid}: {e}")

            # ==================================================
            # Save metadata
            # ==================================================
            meta_path = os.path.join(subject_out, "metadata.json")
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save metadata for {subid}: {e}")

            # Record subject summary
            record.update({
                "status": "OK",
                "n_epochs": int(epochs.shape[0]) if hasattr(epochs, "shape") else 0,
                "pct_bad_segments": float(np.mean(bad_mask)) if hasattr(bad_mask, "__len__") else 0.0,
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
    try:
        summary_df.to_csv(summary_path, index=False)
        logger.success(f"📜 Phase 1 summary saved → {summary_path}")
    except Exception as e:
        logger.warning(f"Could not save phase1 summary CSV: {e}")

    status("Phase 1 completed successfully.")


# =====================================================================
# Phase 2 — Feature Extraction wrapper
# =====================================================================
def run_phase2_from_main(cfg: dict, processed_dir: str, results_dir: str):
    processed_dir, results_dir = os.path.abspath(processed_dir), os.path.abspath(results_dir)
    if not os.path.exists(processed_dir):
        logger.error(f"Processed directory missing: {processed_dir}")
        return
    extract_features_for_cohort(processed_dir, results_dir)
    logger.success("✅ Phase 2: Feature extraction completed successfully.")
