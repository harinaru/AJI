import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# SUBJECT-LEVEL UNCERTAINTY AGGREGATION
# ------------------------------------------------------------------

def compute_subject_uncertainty(
    df: pd.DataFrame,
    uncertainty_col: str = "epoch_uncertainty",
    agg: str = "mean"
):
    """
    Aggregate epoch-level uncertainty to subject-level.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: subject, label, epoch_uncertainty
    agg : str
        Aggregation strategy:
        - "mean"  (primary, paper)
        - "max"   (worst-case)
        - "p90"   (robust high uncertainty)

    Returns
    -------
    subj_df : DataFrame
        subject | label | uncertainty
    """

    if agg == "mean":
        func = np.mean
    elif agg == "max":
        func = np.max
    elif agg == "p90":
        func = lambda x: np.percentile(x, 90)
    else:
        raise ValueError(f"Unknown aggregation: {agg}")

    subj_df = (
        df.groupby("subject")
        .agg({
            uncertainty_col: func,
            "label": "first"
        })
        .reset_index()
        .rename(columns={uncertainty_col: "uncertainty"})
    )

    return subj_df


# ------------------------------------------------------------------
# THRESHOLD SELECTION (CONTROL-ONLY)
# ------------------------------------------------------------------

def compute_threshold(
    control_uncertainties: np.ndarray,
    percentile: int = 95
):
    """
    Compute abstention threshold τ using ONLY control subjects.

    Parameters
    ----------
    control_uncertainties : array-like
        Subject-level uncertainty for controls only
    percentile : int
        Typically 90, 95, 97.5, or 99

    Returns
    -------
    tau : float
        Abstention threshold
    """

    if len(control_uncertainties) == 0:
        raise ValueError("No control uncertainties provided")

    tau = np.percentile(control_uncertainties, percentile)

    return float(tau)


# ------------------------------------------------------------------
# ABSTENTION DECISION
# ------------------------------------------------------------------

def apply_abstention_policy(
    subj_df: pd.DataFrame,
    tau: float
):
    """
    Apply abstention decision based on threshold.

    Adds:
        - committed (True/False)
        - abstained (True/False)
    """

    subj_df = subj_df.copy()
    subj_df["committed"] = subj_df["uncertainty"] < tau
    subj_df["abstained"] = ~subj_df["committed"]

    return subj_df
