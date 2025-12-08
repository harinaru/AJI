# src/phase2_features/feature_manager.py
# v4.4.1_research_final_fix — Pyright-clean CAP-integrated Phase 2 (patched)
"""
Phase 2 Feature Manager (updated to create 3 CSVs per subject with CAP embedding)
- Produces: subj_B183.csv (step=5s), subj_B150.csv (step=15s), subj_B10.csv (step=30s)
- Embeds per-epoch CAP micro features and global Terzano metrics in each CSV
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sps
from loguru import logger
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===============================================================
# Safe imports (unchanged semantics)
# ===============================================================
try:
    from phase2_features.time_domain import extract_time_features_with_metadata
except Exception as e:
    logger.warning(f"[Fallback] time_domain import failed: {e}")
    def extract_time_features_with_metadata(epoch, sfreq, ch_names, epoch_idx, start_s, end_s, overlap_s):
        # ensure numeric fallback
        arr = np.asarray(epoch, dtype=float)
        return {"mean_signal": float(np.nanmean(arr))}

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
    def extract_cap_features(epoch_idx, epoch_times, epoch_labels, cap_events=None, cap_metrics=None):
        return {"cap_present": 0, "cap_subtype": "NA", "cap_a_ratio": 0.0,
                "cap_A_start": np.nan, "cap_A_duration": np.nan}

try:
    from phase1_preprocessing.scoring_reader import read_remlogic_txt
except Exception:
    read_remlogic_txt = None

try:
    from phase1_preprocessing.cap_metrics import compute_terzano_metrics
except Exception:
    compute_terzano_metrics = None

try:
    from phase1_preprocessing.epoching import make_epochs
except Exception:
    make_epochs = None

# ===============================================================
FEATURES_ROOT = os.path.join("results", "phase2_features")
DIAG_DIRNAME = "diagnostics"
# ===============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_np_load(path: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)

# ===============================================================
# Diagnostics helper (accepts ndarray or general sequence)
# ===============================================================
from typing import Any
def diagnostics_psd_heatmap(epoch_array: Any, sfreq: float, out_dir: str, subj: str, ch_idx: int = 0):

    ensure_dir(out_dir)
    try:
        psd_stack = []
        freqs = None
        # allow numpy arrays or lists of arrays
        n_show = min(len(epoch_array), 100) if hasattr(epoch_array, "__len__") else 0
        for i in range(n_show):
            item = epoch_array[i]
            arr = np.asarray(item)
            # pick channel safely
            if arr.ndim == 2:
                if ch_idx < arr.shape[0]:
                    ch = np.asarray(arr[ch_idx], dtype=float)
                else:
                    ch = np.asarray(arr[0], dtype=float)
            else:
                ch = np.asarray(arr, dtype=float)
            if ch.size <= 1:
                continue
            f, p = sps.welch(ch, fs=float(sfreq), nperseg=max(2, int(sfreq * 2)))
            freqs = f
            psd_stack.append(p)
        if not psd_stack or freqs is None:
            return
        psd_stack = np.array(psd_stack)
        plt.figure(figsize=(10, 5))
        plt.imshow(10 * np.log10(psd_stack + 1e-12), aspect="auto", origin="lower",
                   extent=(0, len(psd_stack), float(freqs[0]), float(freqs[-1])), cmap="magma")
        plt.colorbar(label="PSD (dB)")
        plt.title(f"{subj} - PSD Heatmap (ch={ch_idx})")
        plt.xlabel("Epoch Index")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{subj}_psd_heatmap.png"), dpi=250)
        plt.close()
    except Exception as e:
        logger.debug(f"PSD heatmap failed: {e}")

# ===============================================================
# Processed subject loader
# ===============================================================
def load_processed_subject(processed_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(processed_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json missing in {processed_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    data: Dict[str, Any] = {"metadata": meta}
    filtered_path = os.path.join(processed_dir, "filtered.npy")
    if os.path.exists(filtered_path):
        try:
            data["filtered"] = safe_np_load(filtered_path)
        except Exception as e:
            logger.debug(f"safe_np_load failed for filtered.npy: {e}")
            data["filtered"] = None
    else:
        logger.warning("filtered.npy not found; fallback to epochs if available")
        ep_npy = os.path.join(processed_dir, "epochs.npy")
        if os.path.exists(ep_npy):
            try:
                data["epochs"] = safe_np_load(ep_npy)
            except Exception as e:
                logger.debug(f"safe_np_load failed for epochs.npy: {e}")
                data["epochs"] = None

    epoch_times = os.path.join(processed_dir, "epoch_times.npy")
    epoch_labels = os.path.join(processed_dir, "epoch_labels.npy")
    data["epoch_times"] = safe_np_load(epoch_times) if os.path.exists(epoch_times) else None
    data["epoch_labels"] = safe_np_load(epoch_labels) if os.path.exists(epoch_labels) else None

    txt_candidate = os.path.join(os.path.dirname(processed_dir), os.path.basename(processed_dir) + ".txt")
    st_candidate = os.path.join(os.path.dirname(processed_dir), os.path.basename(processed_dir) + ".edf.st")
    if os.path.exists(txt_candidate) and read_remlogic_txt:
        try:
            hyp, ttot, dur, typ = read_remlogic_txt(
                txt_candidate,
                output_json=os.path.join(processed_dir, f"{os.path.basename(processed_dir)}_cap_annotations.json"))
            data["remlogic_hyp"] = hyp
            data["remlogic_time_tot"] = ttot
            data["remlogic_duration"] = dur
            data["remlogic_type_ar"] = typ
        except Exception as e:
            logger.debug(f"scoring reader failed: {e}")
    elif os.path.exists(st_candidate):
        # optionally handle .st files in future
        pass

    data["sfreq"] = float(meta.get("sfreq", 200.0))
    data["ch_names"] = meta.get("ch_names", [])
    data["epoch_length_s"] = float(meta.get("epoch_length_s", 30.0))
    data["epoch_step_s"] = float(meta.get("epoch_step_s", 5.0))
    return data

# ===============================================================
# Compression utilities
# ===============================================================
def compress_to_8bit(epochs: np.ndarray, save_path: str):
    try:
        arr = np.asarray(epochs, dtype=float)
        minv, maxv = float(np.min(arr)), float(np.max(arr))
        scale = (maxv - minv) or 1.0
        scaled = np.clip((arr - minv) / scale * 255, 0, 255).astype(np.uint8)
        np.savez_compressed(save_path, epochs=scaled, min_val=minv, max_val=maxv)
    except Exception as e:
        logger.warning(f"Compression failed: {e}")

def delete_heavy_files(subject_dir: str):
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
        logger.info(f"Deleted heavy files: {deleted}")

# ===============================================================
# Worker
# ===============================================================
def _extract_features_for_epoch_worker(args):
    (epoch, sfreq, ch_names, idx, times, labels, length_s, step_s, cap_events, cap_metrics) = args
    feats: Dict[str, Any] = {}
    try:
        start_s = float(times[idx, 0]) if (times is not None and len(times) > idx) else float(idx * step_s)
    except Exception:
        start_s = float(idx * step_s)
    end_s = start_s + float(length_s)
    overlap = max(0.0, float(length_s) - float(step_s))

    # Ensure numeric array and cast to float to avoid complex/object typing
    arr_epoch = np.asarray(epoch, dtype=float)
    try:
        feats.update(extract_time_features_with_metadata(arr_epoch, float(sfreq), ch_names, int(idx), float(start_s), float(end_s), float(overlap)))
    except Exception:
        feats["mean_signal"] = float(np.nanmean(arr_epoch))

    try:
        # safe call: external function may expect 1d; pass 1d if single channel else multi
        feats.update(extract_freq_features(arr_epoch, float(sfreq)) if callable(extract_freq_features) else {})
    except Exception:
        pass
    try:
        feats.update(extract_wavelet_energy(arr_epoch, float(sfreq)) if callable(extract_wavelet_energy) else {})
    except Exception:
        pass

    try:
        cap_feats = extract_cap_features(int(idx), times, labels, cap_events=cap_events, cap_metrics=cap_metrics)
        feats.update(cap_feats or {})
    except Exception:
        feats.update({"cap_present": 0, "cap_subtype": "NA", "cap_a_ratio": 0.0,
                      "cap_A_start": np.nan, "cap_A_duration": np.nan})

    feats.update({
        "epoch_idx": int(idx),
        "epoch_start_s": float(start_s),
        "epoch_end_s": float(end_s),
        "epoch_overlap_s": float(overlap),
    })
    return feats

# ===============================================================
# Batch runner
# ===============================================================
def _run_epoch_batch_and_save(epochs, sfreq, ch_names, times, labels,
                              cap_events, cap_metrics,
                              length_s, step_s, out_csv_path, subj, diag_dir):
    ensure_dir(os.path.dirname(out_csv_path))
    args_list = [(ep, sfreq, ch_names, i, times, labels, length_s, step_s, cap_events, cap_metrics)
                 for i, ep in enumerate(epochs)]
    records: List[Dict[str, Any]] = []
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_features_for_epoch_worker, a) for a in args_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Features {subj}"):
            try:
                rec = fut.result()
                if rec:
                    records.append(rec)
            except Exception as e:
                logger.debug(f"Epoch fail: {e}")

    if not records:
        logger.warning(f"No records extracted for {subj} -> {out_csv_path}")
        return False

    df = pd.DataFrame.from_records(records)

    # ✅ Flatten and attach CAP metrics safely
    if cap_metrics:
        for k, v in cap_metrics.items():
            if isinstance(v, (int, float, np.number)):
                df[f"cap_{k}"] = float(v)
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if isinstance(subv, (int, float, np.number)):
                        df[f"cap_{k}_{subk}"] = float(subv)
                    else:
                        df[f"cap_{k}_{subk}"] = np.nan
            else:
                df[f"cap_{k}"] = np.nan

    df.to_csv(out_csv_path, index=False)
    logger.info(f"Saved features CSV -> {out_csv_path}")

    try:
        diagnostics_psd_heatmap(np.array(epochs, dtype=object), float(sfreq), diag_dir, subj, ch_idx=0)
    except Exception:
        pass
    return True

# ===============================================================
# Subject-level extraction
# ===============================================================
def extract_features_for_subject(subject_dir: str, results_root: str) -> Optional[List[str]]:
    subj = os.path.basename(subject_dir.rstrip("/\\"))  # keep windows/backslash safe
    logger.info(f"Extracting features for {subj}")
    last_epochs_for_compress = None

    try:
        data = load_processed_subject(subject_dir)
    except Exception as e:
        logger.error(f"Load failed {subj}: {e}")
        return None

    sfreq = float(data.get("sfreq", 200.0))
    ch_names = data.get("ch_names", [])
    length_s = 30.0
    configs = [("B183", 5), ("B150", 15), ("B10", 30)]

    cap_events, cap_metrics = None, None
    cap_json = os.path.join(subject_dir, f"{subj}_cap_annotations.json")
    cap_metrics_json = os.path.join(subject_dir, f"{subj}_cap_metrics.json")

    if os.path.exists(cap_json):
        try:
            with open(cap_json, "r", encoding="utf-8") as f:
                cap_events = json.load(f)
        except Exception:
            cap_events = None

    if os.path.exists(cap_metrics_json):
        try:
            with open(cap_metrics_json, "r", encoding="utf-8") as f:
                cap_metrics = json.load(f)
        except Exception:
            cap_metrics = None

    filtered = data.get("filtered", None)
    if filtered is None:
        if "epochs" in data and data.get("epochs") is not None:
            logger.warning("filtered.npy missing; using precomputed epochs (single overlap only).")
            epochs = data["epochs"]
            out_dir = os.path.join(results_root, subj)
            ensure_dir(out_dir)
            diag_dir = os.path.join(out_dir, DIAG_DIRNAME); ensure_dir(diag_dir)
            csv_path = os.path.join(out_dir, f"{subj}_epoch_features.csv")
            _run_epoch_batch_and_save(epochs, sfreq, ch_names, data.get("epoch_times"),
                                      data.get("epoch_labels"), cap_events, cap_metrics,
                                      length_s, data.get("epoch_step_s", 5),
                                      csv_path, subj, diag_dir)
            # set compression fallback
            last_epochs_for_compress = np.asarray(epochs, dtype=object)
            return [csv_path]
        else:
            logger.error("No filtered or epoch data — cannot proceed.")
            return None

    saved_csvs: List[str] = []
    # normalize filtered to numeric ndarray
    try:
        filtered_arr = np.asarray(filtered)
        # if 1D make it 2D channels x samples
        if filtered_arr.ndim == 1:
            filtered_arr = filtered_arr[np.newaxis, :]
        elif filtered_arr.ndim > 2:
            # try to reshape to 2D if possible (channels, samples)
            filtered_arr = filtered_arr.reshape((filtered_arr.shape[0], -1))
    except Exception as e:
        logger.error(f"Failed to normalize filtered array for {subj}: {e}")
        return None

    for suffix, step_s in configs:
        if make_epochs is None:
            logger.error("make_epochs missing — check phase1_preprocessing import.")
            return None

        # try to use make_epochs if available (keeps previous behavior)
        try:
            epochs, epoch_times = make_epochs(filtered_arr, sfreq,
                                              epoch_length_s=int(length_s),
                                              step_s=int(step_s),
                                              bad_mask=None)
        except Exception:
            # fallback: manual sliding window
            length_samples = int(length_s * sfreq)
            step_samples = int(step_s * sfreq)
            n_samples = filtered_arr.shape[1]
            epochs_list = []
            times_list = []
            start = 0
            while start + length_samples <= n_samples:
                ep = filtered_arr[:, start:start + length_samples]
                epochs_list.append(ep.astype(float))
                times_list.append((start / sfreq, (start + length_samples) / sfreq))
                start += step_samples
            epochs = epochs_list
            epoch_times = np.array(times_list, dtype=float)

        # ensure epoch_times is array of [start,end]
        if isinstance(epoch_times, (list, tuple)) and epoch_times and isinstance(epoch_times[0], dict):
            times_arr = np.array([[et.get("start", 0.0), et.get("end", 0.0)] for et in epoch_times], dtype=float)
        elif isinstance(epoch_times, np.ndarray):
            times_arr = epoch_times
        else:
            times_arr = np.array([[i * step_s, i * step_s + length_s] for i in range(len(epochs))], dtype=float)

        labels = data.get("epoch_labels", None)
        out_dir = os.path.join(results_root, subj); ensure_dir(out_dir)
        diag_dir = os.path.join(out_dir, DIAG_DIRNAME); ensure_dir(diag_dir)
        csv_name = f"{subj}_{suffix}.csv"
        csv_path = os.path.join(out_dir, csv_name)

        ok = _run_epoch_batch_and_save(epochs, sfreq, ch_names, times_arr, labels,
                                       cap_events, cap_metrics, length_s, step_s,
                                       csv_path, subj, diag_dir)
        if ok:
            saved_csvs.append(csv_path)
            # keep last produced epochs for compression/cleanup
            try:
                last_epochs_for_compress = np.asarray(epochs, dtype=object)
            except Exception:
                last_epochs_for_compress = None

    # ===============================================================
    # Final cleanup + compression
    # ===============================================================
    try:
        if last_epochs_for_compress is not None:
            epochs_to_compress = np.asarray(last_epochs_for_compress, dtype=float)
            if epochs_to_compress.size > 0:
                compress_to_8bit(epochs_to_compress, os.path.join(subject_dir, "epochs_8bit.npz"))
                logger.info(f"🗜️ Compressed epochs -> {os.path.join(subject_dir, 'epochs_8bit.npz')}")
        elif filtered_arr is not None:
            filtered_arr_float = np.asarray(filtered_arr, dtype=float)
            if filtered_arr_float.size > 0:
                compress_to_8bit(filtered_arr_float, os.path.join(subject_dir, "filtered_8bit.npz"))
                logger.info(f"🗜️ Compressed filtered -> {os.path.join(subject_dir, 'filtered_8bit.npz')}")
        delete_heavy_files(subject_dir)
        logger.info(f"🧹 Cleanup complete for {subj}")
    except Exception as e:
        logger.debug(f"Cleanup skipped for {subj}: {e}")

    return saved_csvs

# ===============================================================
# Cohort runner
# ===============================================================
def extract_features_for_cohort(processed_root="data/processed", results_root=FEATURES_ROOT):
    ensure_dir(results_root)
    subjects = [d for d in sorted(os.listdir(processed_root))
                if os.path.isdir(os.path.join(processed_root, d))]
    for subj in subjects:
        extract_features_for_subject(os.path.join(processed_root, subj), results_root)
    logger.success("✅ Phase 2 complete — all subjects processed.")
