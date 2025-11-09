# src/phase2_features/nonlinear_features.py
import numpy as np
# use antropy and nolds if available; otherwise simple fallbacks
try:
    import antropy as ant
except Exception:
    ant = None
try:
    import nolds
except Exception:
    nolds = None

def hurst_exponent(x):
    if nolds is not None:
        try:
            return float(nolds.hurst_rs(x))
        except Exception:
            pass
    # fallback: approximate by 0.5
    return 0.5

def sample_entropy(x):
    if ant is not None:
        try:
            return float(ant.sample_entropy(x))
        except Exception:
            pass
    # fallback: small constant
    return 0.0

def fractal_katz(x):
    try:
        # simple Katz fractal dimension
        L = np.sum(np.abs(np.diff(x)))
        d = len(x)
        R = np.max(np.abs(x - x[0]))
        if R == 0 or L == 0:
            return 0.0
        return float(np.log10(L / R) / np.log10(d))
    except Exception:
        return 0.0

def extract_nonlinear(epoch):
    feats = {}
    if epoch.ndim == 2:
        x = np.mean(epoch, axis=0)
    else:
        x = epoch
    feats['hurst'] = hurst_exponent(x)
    feats['samp_entropy'] = sample_entropy(x)
    feats['fractal_katz'] = fractal_katz(x)
    return feats
