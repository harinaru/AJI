# ===============================================================
# src/core/pipeline_controller.py
# v4.3.7_stable — Pyright-safe & fully Phase 2 compatible
# ===============================================================
import os
import json
import traceback
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Utilities and core modules
from utils.folder_creation import verify_all
from utils.logger import get_logger
from utils.progress_tracker import status
from phase1_preprocessing.data_loader import discover_subjects,load_subject_normalized
from phase1_preprocessing.filtering import apply_filters, detect_artifacts_threshold
from phase1_preprocessing.epoching import make_epochs, label_epochs, save_processed_subject
from phase1_preprocessing.st_annotation_parser import parse_st_file, map_caps_to_epochs
from phase2_features.feature_manager import extract_features_for_cohort

# Reproducibility
SEED = 42
np.random.seed(SEED)


# =====================================================================
# 🧩 Phase 1: Preprocessing
# =====================================================================
def run_phase1_from_main(cfg: Dict[str, Any],
                         raw_dir: str,
                         out_dir: str,
                         subjects: Optional[List[str]] = None,
                         workers: int = 1,
                         debug: bool = False) -> None:
    """
    Phase 1 preprocessing pipeline:
      - Verify folder structure
      - Load and normalize subject data
      - Apply filtering and artifact detection
      - Segment into epochs
      - Label epochs and parse CAP annotations
      - Save processed data and metadata for Phase 2
    """

    if debug:
        logger.info("⚙️ Debug mode enabled")

    logger.info("🧩 Phase 1: Verifying folder structure...")
    verify_all()

    raw_dir = os.path.abspath(raw_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Discover subjects
    # ------------------------------------------------------------------
    logger.info(f"📂 Scanning raw data directory: {raw_dir}")
    all_subjects = discover_subjects(raw_dir)
    subject_list = [s for s in all_subjects if s in (subjects or all_subjects)]

    if not subject_list:
        logger.warning("⚠️ No subjects found to process — check input folder.")
        return

    summary_records: List[Dict[str, Any]] = []
    status(f"Processing {len(subject_list)} subjects...")

    # ------------------------------------------------------------------
    # Process subjects
    # ------------------------------------------------------------------
    for subid in tqdm(subject_list, desc="Processing subjects..."):
        if subid.endswith(".st"):
            logger.info(f"🛈 Skipping annotation-only file: {subid}")
            continue

        record: Dict[str, Any] = {"subject": subid}

        try:
            # ==========================================================
            # Load normalized subject
            # ==========================================================
            subj = load_subject_normalized(subid, raw_dir)
            data = subj["data"]
            sfreq = float(subj["sfreq"])
            ch_names = subj.get("ch_names", [])
            annotations = subj.get("annotations", [])

            # ==========================================================
            # Filtering
            # ==========================================================
            filtered, filt_meta = apply_filters(
                data,
                sfreq,
                low=float(cfg.get("filter_low", 0.3)),
                high=float(cfg.get("filter_high", 35.0)),
                notch_freqs=cfg.get("notch_freqs", [50.0])
            )

            # ==========================================================
            # Artifact detection
            # ==========================================================
            bad_mask = detect_artifacts_threshold(
                filtered,
                sfreq,
                thresh_uV=float(cfg.get("artifact_thresh_uV", 200.0))
            )

            # ==========================================================
            # Epoching
            # ==========================================================
            epoch_len = int(float(cfg.get("epoch_length_s", 30)))
            step_s = int(float(cfg.get("epoch_step_s", 5)))

            epochs, epoch_times = make_epochs(
                filtered,
                sfreq,
                epoch_length_s=epoch_len,
                step_s=step_s,
                bad_mask=bad_mask
            )

            # ==========================================================
            # Label epochs
            # ==========================================================
            epoch_labels = label_epochs(
                epoch_times,
                annotations,
                horizon_s=int(float(cfg.get("prediction_horizon_s", 30)))
            )

            # ==========================================================
            # Save processed data + metadata
            # ==========================================================
            subject_out = os.path.join(out_dir, subid)
            os.makedirs(subject_out, exist_ok=True)

            save_processed_subject(
                subject_out,
                subid,
                filtered,
                sfreq,
                ch_names,
                epochs,
                epoch_times,
                epoch_labels,
                annotations
            )

            # Save metadata in Phase-2 compatible format
            meta = {
                "subject": subid,
                "source_path": subj.get("source_path"),
                "n_channels": int(filtered.shape[0]),
                "n_samples": int(filtered.shape[1]),
                "sfreq": sfreq,
                "ch_names": ch_names,
                "filter_meta": filt_meta,
                "epoch_length_s": epoch_len,
                "epoch_step_s": step_s,
                "seed": SEED
            }

            meta_path = os.path.join(subject_out, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            logger.success(f"✅ Saved processed subject {subid} ({epochs.shape[0]} epochs)")

            # ==========================================================
            # CAP Annotation Parsing (.st)
            # ==========================================================
            annotation_path = os.path.join(raw_dir, f"{subid}.edf.st")
            if os.path.exists(annotation_path):
                try:
                    logger.info(f"📄 Found annotation file for {subid} → parsing CAP events...")
                    cap_json_path = os.path.join(subject_out, f"{subid}_cap_annotations.json")
                    cap_annotations = parse_st_file(annotation_path, output_json=cap_json_path)

                    epoch_times_path = os.path.join(subject_out, "epoch_times.npy")
                    if os.path.exists(epoch_times_path):
                        epoch_times_loaded = np.load(epoch_times_path, allow_pickle=True)
                        epoch_labels_cap = map_caps_to_epochs(
                            cap_annotations,
                            epoch_times_loaded,
                            epoch_step_s=step_s,
                            n_epochs=len(epoch_times_loaded)
                        )
                        np.save(os.path.join(subject_out, "epoch_labels.npy"), epoch_labels_cap)
                        logger.success(
                            f"✅ CAP epoch labels generated for {subid} ({len(epoch_labels_cap)} epochs)"
                        )
                    else:
                        logger.warning(f"⚠️ Missing epoch_times.npy — skipping CAP label mapping for {subid}")
                except Exception as e:
                    logger.error(f"❌ Failed to parse CAP annotations for {subid}: {e}")
            else:
                logger.debug(f"No annotation (.st) file found for {subid}")

            # ==========================================================
            # Summary — Pyright-safe explicit dict assignment
            # ==========================================================
            n_epochs = int(epochs.shape[0]) if epochs is not None else 0
            pct_bad = float(np.mean(bad_mask)) if bad_mask is not None else 0.0
            record["status"] = "OK"
            record["n_epochs"] = n_epochs
            record["pct_bad_segments"] = pct_bad
            record["message"] = ""

            logger.info(f"✅ Completed {subid}: {n_epochs} epochs | bad={pct_bad:.3f}")

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"❌ Error processing {subid}: {e}\n{tb}")
            record["status"] = "FAILED"
            record["n_epochs"] = 0
            record["pct_bad_segments"] = 0.0
            record["message"] = str(e)
        finally:
            summary_records.append(record)

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join("results", "phase1_preprocessing", "tables", "processing_summary.csv")
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    logger.success(f"📜 Phase 1 completed — Summary written to {summary_csv}")
    status("Phase 1 completed successfully.")


# =====================================================================
# 🧠 Phase 2: Feature Extraction
# =====================================================================
def run_phase2_from_main(cfg: dict, processed_dir: str, results_dir: str):
    """
    Phase 2: Extract per-epoch features (time, freq, wavelet, CAP)
    """
    logger.info("Phase2: Verifying inputs...")
    processed_dir = os.path.abspath(processed_dir)
    results_dir = os.path.abspath(results_dir)

    if not os.path.exists(processed_dir):
        logger.error(f"Processed directory does not exist: {processed_dir}")
        return

    logger.info(f"Phase2: Scanning processed dir: {processed_dir}")
    extract_features_for_cohort(processed_dir, results_dir)
    logger.success("✅ Phase 2: Completed successfully.")
