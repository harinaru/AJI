import os
import json
import numpy as np
from loguru import logger


# ================================================================
# 🧩 Epoch Creation Function
# ================================================================
def make_epochs(data, sfreq, epoch_length_s=30, step_s=5, bad_mask=None):
    """
    Segment continuous EEG data into overlapping epochs.

    Args:
        data: ndarray (n_channels, n_samples)
        sfreq: sampling frequency (Hz)
        epoch_length_s: window length in seconds
        step_s: step size in seconds
        bad_mask: optional boolean mask for artifact segments

    Returns:
        epochs: ndarray (n_epochs, n_channels, n_samples_per_epoch)
        epoch_times: list of dicts [{'start': t0, 'end': t1}, ...]
    """
    n_channels, n_samples = data.shape
    samples_per_epoch = int(epoch_length_s * sfreq)
    step_samples = int(step_s * sfreq)
    epoch_list, times_list = [], []

    for start_idx in range(0, n_samples - samples_per_epoch + 1, step_samples):
        end_idx = start_idx + samples_per_epoch
        seg = data[:, start_idx:end_idx]

        # Skip if bad_mask marks this region
        if bad_mask is not None:
            segment_mask = bad_mask[start_idx:end_idx]
            if np.any(segment_mask):
                continue

        epoch_list.append(seg)
        times_list.append({
            "start": start_idx / sfreq,
            "end": end_idx / sfreq
        })

    if not epoch_list:
        logger.warning("⚠️ No valid epochs created (possibly due to heavy artifact masking).")
        return np.empty((0, n_channels, samples_per_epoch)), []

    epochs = np.stack(epoch_list, axis=0)
    logger.info(f"✅ Created {len(epochs)} epochs | shape={epochs.shape}")
    return epochs, times_list


# ================================================================
# 🏷️ Epoch Labeling Function
# ================================================================
def label_epochs(epoch_times, annotations, horizon_s=30):
    """
    Label each epoch using overlap with provided annotations (.st).

    Args:
        epoch_times: list of dicts [{'start':, 'end':}]
        annotations: list of dicts [{'start':, 'end':, 'label':}]
        horizon_s: time window threshold for overlap

    Returns:
        np.ndarray of integer labels per epoch
    """
    labels = np.zeros(len(epoch_times), dtype=int)

    if not annotations:
        logger.debug("No annotations provided — labeling all epochs as 0.")
        return labels

    for i, et in enumerate(epoch_times):
        e_start, e_end = et["start"], et["end"]
        for ann in annotations:
            a_start, a_end = ann.get("start", 0), ann.get("end", 0)
            overlap = max(0, min(e_end, a_end) - max(e_start, a_start))
            if overlap >= min(horizon_s, (e_end - e_start) * 0.25):
                labels[i] = 1  # Mark as event/positive epoch
                break

    logger.info(f"✅ Labeled {np.sum(labels)} / {len(labels)} epochs as positive events.")
    return labels


# ================================================================
# 💾 Save Processed Subject (your existing version)
# ================================================================
def save_processed_subject(out_dir, subid, filtered, sfreq, ch_names,
                           epochs, epoch_times, epoch_labels, annotations):
    """Save processed EEG data, epoch metadata, and related info safely."""
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 🧩 Normalize epoch_times before saving
    # ------------------------------------------------------------------
    epoch_times_arr = None
    try:
        if isinstance(epoch_times, (list, tuple)) and len(epoch_times) > 0:
            first = epoch_times[0]
            if isinstance(first, dict):
                # Extract 'start' and 'end' keys if dicts
                epoch_times_arr = np.array(
                    [[float(et.get("start", 0)), float(et.get("end", 0))]
                     for et in epoch_times],
                    dtype=np.float32,
                )
            elif isinstance(first, (list, tuple)) and len(first) >= 2:
                epoch_times_arr = np.array(epoch_times, dtype=np.float32)
            else:
                epoch_times_arr = np.expand_dims(np.asarray(epoch_times, dtype=np.float32), axis=1)
        else:
            epoch_times_arr = np.empty((0, 2), dtype=np.float32)
    except Exception as e:
        logger.warning(f"⚠️ epoch_times conversion failed for {subid}: {e}")
        epoch_times_arr = np.empty((0, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # 💾 Save core arrays
    # ------------------------------------------------------------------
    np.save(os.path.join(out_dir, "filtered.npy"), filtered.astype(np.float32))
    np.save(os.path.join(out_dir, "epochs.npy"), epochs.astype(np.float32))
    np.save(os.path.join(out_dir, "epoch_times.npy"), epoch_times_arr)
    if epoch_labels is not None:
        np.save(os.path.join(out_dir, "epoch_labels.npy"), np.asarray(epoch_labels, dtype=np.int32))

    # ------------------------------------------------------------------
    # 🧠 Save metadata
    # ------------------------------------------------------------------
    meta = {
        "subject": subid,
        "source_path": os.path.join("data", "raw", f"{subid}.edf"),
        "n_channels": int(filtered.shape[0]),
        "n_samples": int(filtered.shape[1]),
        "sfreq": float(sfreq),
        "ch_names": ch_names,
        "epoch_length_s": float(30.0),
        "epoch_step_s": float(5.0),
        "filter_meta": {
            "low_cutoff": 0.3,
            "high_cutoff": 35.0,
            "notch_freqs": [50.0]
        },
        "annotations": len(annotations) if annotations else 0
    }

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.success(f"✅ Saved processed subject {subid} → {out_dir}")
