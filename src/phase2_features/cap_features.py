# src/phase2_features/cap_features.py
import numpy as np

def extract_cap_features(epoch_idx, epoch_times, epoch_labels):
    """
    epoch_idx: index of current epoch
    epoch_times: numpy array shape (n_epochs,2) (start_s, end_s)
    epoch_labels: can be:
      - per-epoch label array (n_epochs,) with A1/A2/A3/B or None
      - or higher-resolution annotations if available (not covered here)
    This function returns CAP related features for the epoch.
    """
    feats = {}
    if epoch_labels is None:
        feats['cap_present'] = 0
        feats['cap_subtype'] = 'NA'
        feats['cap_a_ratio'] = 0.0
    else:
        lab = epoch_labels[epoch_idx]
        # If label is a string like 'A1' treat as present
        if isinstance(lab, (str,)):
            feats['cap_present'] = 1 if lab.upper().startswith('A') else 0
            feats['cap_subtype'] = lab
            feats['cap_a_ratio'] = 1.0 if feats['cap_present'] else 0.0
        else:
            # numeric labels fallback
            feats['cap_present'] = int(bool(lab))
            feats['cap_subtype'] = str(lab)
            feats['cap_a_ratio'] = float(lab)
    return feats
