import pandas as pd
from typing import List
from .channel_config import COMMON_COLUMNS


def _normalize(name: str) -> str:
    """
    Normalize names for robust channel matching.
    Example:
        Fp2-F4_mean  -> fp2f4mean
        Fp2_F4_mean  -> fp2f4mean
    """
    return name.lower().replace("-", "").replace("_", "")


def extract_channel_df(
    df: pd.DataFrame,
    channel: str,
    subject_id: str
) -> pd.DataFrame:
    """
    Extract one channel's features + common columns
    and attach subject identity.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataframe for one subject (all channels)
    channel : str
        Channel name (e.g., 'Fp2-F4')
    subject_id : str
        Subject identifier (e.g., 'brux1', 'ins2')

    Returns
    -------
    pd.DataFrame
        Channel-specific dataframe with:
        - subject
        - common (epoch/global/CAP) columns
        - channel-specific feature columns
    """

    # Normalize channel name once
    channel_key = _normalize(channel)

    # 1️⃣ Robust channel-specific column detection
    channel_cols: List[str] = [
        c for c in df.columns
        if _normalize(c).startswith(channel_key)
    ]

    if not channel_cols:
        raise ValueError(f"No columns found for channel {channel}")

    # 2️⃣ Keep only common columns that actually exist
    common_cols: List[str] = [
        c for c in COMMON_COLUMNS if c in df.columns
    ]

    # 3️⃣ Build output dataframe
    out_df = df.loc[:, common_cols + channel_cols].copy()

    # 4️⃣ Attach subject identity explicitly
    out_df.insert(0, "subject", subject_id)

    # 5️⃣ Final cleanup
    return out_df.reset_index(drop=True)
