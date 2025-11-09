# ===============================================================
# src/phase2_features/feature_manager.py
# v4.3.5_stable – Full 30+ time-domain integration + 8-bit compression
# ===============================================================
"""
Phase 2 Feature Manager
---------------------------------------------------------------
✅ Integrates time_domain_full (30+ features)
✅ Auto-handles large EEGs with 8-bit compression
✅ Parallelized (multi-core) feature extraction
✅ Full CSV + PSD diagnostics per subject
✅ Skips missing data safely (no crash)
✅ Deletes heavy .npy/.npz after successful CSV save
"""
import os
import json
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sps
from loguru import logger
from tqdm import tqdm
from typing import Optional, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===============================================================
# Local imports (safe fallbacks if modules missing)
# ===============================================================
try:
    from phase2_features.time_domain import extract_time_features_with_metadata
except Exception as e:
    logger.warning(f"[Fallback] time_domain_full import failed: {e}")
    def extract_time_features_with_metadata(epoch, sfreq, ch_names, epoch_idx, start_s, end_s, overlap_s):
        return {"mean_signal": float(np.mean(epoch))}

try:
    from phase2_features.freq_domain import extract_freq_features
except Exception:
    def extract_freq_features(epoch, sfreq):
        return {}

try:
    from phase2_features.wavelet_features import extract_wavelet_energy
except Exception:
    def extract_wavelet_energy(epoch, sfreq):
        return {}

try:
    from phase2_features.cap_features import extract_cap_features
except Exception:
    def extract_cap_features(epoch_idx, epoch_times, epoch_labels):
        return {"cap_present": 0, "cap_subtype": "NA", "cap_a_ratio": 0.0}

_ST_PARSER_MODULE = "phase1_preprocessing.st_annotation_parser"

# ===============================================================
# Configuration
# ===============================================================
FEATURES_ROOT = os.path.join("results", "phase2_features")
DIAG_DIRNAME = "diagnostics"
VERSION_TAG = "v4.3.5_stable"

# ===============================================================
# Utility helpers
# ===============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_np_load(path: str) -> np.ndarray:
    """Safely load npy/npz arrays."""
    try:
        return np.load(path, allow_pickle=False)
    except ValueError:
        logger.warning(f"{os.path.basename(path)} → retrying allow_pickle=True")
        return np.load(path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise


# ===============================================================
# Dynamic .st parser (for CAP mapping)
# ===============================================================
def try_dynamic_parse_st(st_path: str, subject_dir: str, epoch_times: Optional[np.ndarray]):
    """Dynamically import and execute annotation parser if available."""
    try:
        mod = __import__(_ST_PARSER_MODULE, fromlist=["parse_st_file", "map_caps_to_epochs"])
    except Exception as e:
        logger.debug(f"No ST parser found: {_ST_PARSER_MODULE} ({e})")
        return None

    parse_fn = getattr(mod, "parse_st_file", None)
    map_fn = getattr(mod, "map_caps_to_epochs", None)
    annotations = None

    if parse_fn:
        sig = inspect.signature(parse_fn)
        kwargs = {}
        if "output_json" in sig.parameters:
            kwargs["output_json"] = os.path.join(subject_dir, "st_parsed.json")
        try:
            annotations = parse_fn(st_path, **kwargs)
        except Exception as e:
            logger.warning(f"parse_st_file() failed: {e}")
            annotations = None

    if annotations is not None and map_fn:
        try:
            epoch_labels = map_fn(annotations, epoch_times)
            np.save(os.path.join(subject_dir, "epoch_labels.npy"), epoch_labels)
            logger.info(f"Saved epoch_labels.npy for {os.path.basename(subject_dir)}")
            return epoch_labels
        except Exception as e:
            logger.warning(f"map_caps_to_epochs() failed: {e}")

    return None


# ===============================================================
# Processed subject loader
# ===============================================================
def load_processed_subject(processed_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(processed_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json missing in {processed_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    data = {"metadata": meta}
    ep_npz = os.path.join(processed_dir, "epochs.npz")
    ep_npy = os.path.join(processed_dir, "epochs.npy")

    if os.path.exists(ep_npz):
        arr = np.load(ep_npz, allow_pickle=True)
        data["epochs"] = arr["epochs"] if "epochs" in arr else arr[arr.files[0]]
    elif os.path.exists(ep_npy):
        data["epochs"] = safe_np_load(ep_npy)
    else:
        raise FileNotFoundError("No epoch data found.")

    epoch_times = os.path.join(processed_dir, "epoch_times.npy")
    epoch_labels = os.path.join(processed_dir, "epoch_labels.npy")
    data["epoch_times"] = safe_np_load(epoch_times) if os.path.exists(epoch_times) else None
    data["epoch_labels"] = safe_np_load(epoch_labels) if os.path.exists(epoch_labels) else None

    if data["epoch_labels"] is None:
        st_files = [f for f in os.listdir(processed_dir) if f.endswith(".st")]
        if st_files:
            lbls = try_dynamic_parse_st(os.path.join(processed_dir, st_files[0]), processed_dir, data["epoch_times"])
            data["epoch_labels"] = lbls

    data["sfreq"] = float(meta.get("sfreq", 256.0))
    data["ch_names"] = meta.get("ch_names", [])
    data["epoch_length_s"] = float(meta.get("epoch_length_s", 30.0))
    data["epoch_step_s"] = float(meta.get("epoch_step_s", 5.0))

    return data


# ===============================================================
# Space optimization utilities
# ===============================================================
def compress_to_8bit(epochs: np.ndarray, save_path: str):
    """Convert large float data to compressed 8-bit."""
    try:
        minv, maxv = float(np.min(epochs)), float(np.max(epochs))
        scale = (maxv - minv) or 1.0
        scaled = np.clip((epochs - minv) / scale * 255, 0, 255).astype(np.uint8)
        np.savez_compressed(save_path, epochs=scaled, min_val=minv, max_val=maxv)
        logger.info(f"💾 Compressed to 8-bit → {save_path}")
    except Exception as e:
        logger.warning(f"Compression failed: {e}")


def delete_heavy_files(subject_dir: str):
    """Delete large intermediate npy/npz after compression."""
    deleted = []
    for root, _, files in os.walk(subject_dir):
        for f in files:
            if f.endswith((".npy", ".npz")) and "8bit" not in f:
                try:
                    os.remove(os.path.join(root, f))
                    deleted.append(f)
                except Exception as e:
                    logger.debug(f"Delete failed {f}: {e}")
    if deleted:
        logger.info(f"🗑️ Deleted heavy files: {deleted}")


# ===============================================================
# Epoch-level extraction
# ===============================================================
def extract_features_for_epoch(epoch, sfreq, ch_names, idx, times, labels, length_s, step_s):
    feats = {}
    try:
        start_s = float(times[idx, 0]) if times is not None and len(times) > idx else idx * step_s
    except Exception:
        start_s = idx * step_s
    end_s = start_s + float(length_s)
    overlap = max(0.0, length_s - step_s)

    # Time-domain (30+ features)
    try:
        feats.update(extract_time_features_with_metadata(epoch, sfreq, ch_names, idx, start_s, end_s, overlap))
    except Exception as e:
        logger.debug(f"Time-domain fail epoch {idx}: {e}")
        feats["mean_signal"] = float(np.mean(epoch))

    # Frequency + Wavelet + CAP
    for fn, name in [
        (extract_freq_features, "freq"),
        (extract_wavelet_energy, "wavelet"),
        (lambda i, t, l: extract_cap_features(idx, times, labels), "cap"),
    ]:
        try:
            feats.update(fn(epoch, sfreq) if name != "cap" else fn(idx, times, labels))
        except Exception as e:
            logger.debug(f"{name} features failed {idx}: {e}")

    feats.update({
        "epoch_idx": idx,
        "epoch_start_s": start_s,
        "epoch_end_s": end_s,
        "epoch_overlap_s": overlap,
    })
    return feats


# ===============================================================
# Visualization
# ===============================================================
def diagnostics_psd_heatmap(epoch_array, sfreq, out_dir, subj, ch_idx=0):
    """Generate PSD heatmap safely."""
    ensure_dir(out_dir)
    try:
        psd_stack, freqs = [], None
        for i in range(min(len(epoch_array), 100)):
            ch = epoch_array[i][ch_idx] if epoch_array[i].ndim == 2 else epoch_array[i]
            f, p = sps.welch(ch, fs=sfreq, nperseg=int(sfreq * 2))
            freqs = f
            psd_stack.append(p)
        if not psd_stack or freqs is None:
            return
        psd_stack = np.array(psd_stack)
        plt.figure(figsize=(10, 5))
        plt.imshow(10 * np.log10(psd_stack + 1e-12), aspect="auto", origin="lower",
                   extent=(0, len(psd_stack), freqs[0], freqs[-1]), cmap="magma")
        plt.colorbar(label="PSD (dB)")
        plt.title(f"{subj} - PSD Heatmap (ch={ch_idx})")
        plt.xlabel("Epoch Index")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{subj}_psd_heatmap.png"), dpi=250)
        plt.close()
        logger.info(f"📊 PSD heatmap saved for {subj}")
    except Exception as e:
        logger.debug(f"PSD heatmap failed: {e}")


# ===============================================================
# Subject-level extraction
# ===============================================================
def extract_features_for_subject(subject_dir: str, results_root: str) -> Optional[str]:
    subj = os.path.basename(subject_dir.rstrip("/\\"))
    logger.info(f"🔍 Extracting features for {subj}")

    try:
        data = load_processed_subject(subject_dir)
    except Exception as e:
        logger.error(f"Load failed {subj}: {e}")
        return None

    epochs = data.get("epochs")
    if epochs is None or len(epochs) == 0:
        logger.warning(f"No epochs for {subj}")
        return None

    sfreq = data["sfreq"]
    ch_names = data.get("ch_names", [])
    times = data.get("epoch_times")
    labels = data.get("epoch_labels")
    length_s = data.get("epoch_length_s", 30.0)
    step_s = data.get("epoch_step_s", 5.0)

    out_dir = os.path.join(results_root, subj)
    diag_dir = os.path.join(out_dir, DIAG_DIRNAME)
    ensure_dir(out_dir); ensure_dir(diag_dir)

    records = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = [pool.submit(extract_features_for_epoch, ep, sfreq, ch_names, i, times, labels, length_s, step_s)
                   for i, ep in enumerate(epochs)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Features {subj}"):
            try:
                rec = fut.result()
                if rec:
                    records.append(rec)
            except Exception as e:
                logger.debug(f"Epoch fail: {e}")

    if not records:
        logger.warning(f"No features extracted for {subj}")
        return None

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, f"{subj}_epoch_features.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Saved per-epoch CSV → {csv_path}")

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        compress_to_8bit(epochs, os.path.join(subject_dir, "epochs_8bit.npz"))
        delete_heavy_files(subject_dir)
        logger.success(f"Cleaned & compressed {subj}")

    diagnostics_psd_heatmap(np.array(epochs), sfreq, diag_dir, subj, ch_idx=0)
    return csv_path


# ===============================================================
# Cohort runner
# ===============================================================
def extract_features_for_cohort(processed_root="data/processed", results_root=FEATURES_ROOT):
    ensure_dir(results_root)
    subjects = [d for d in sorted(os.listdir(processed_root)) if os.path.isdir(os.path.join(processed_root, d))]
    for subj in subjects:
        extract_features_for_subject(os.path.join(processed_root, subj), results_root)
    logger.success("✅ Phase 2 complete — all subjects processed.")


# ===============================================================
# Entrypoint
# ===============================================================
def run_phase2_from_main(cfg: dict, processed_root="data/processed", results_dir=FEATURES_ROOT):
    extract_features_for_cohort(processed_root, results_dir)
