# src/phase7_1_cap_ltc/preprocess/feature_selector.py

"""
Feature selector for Phase 7 preprocessing
------------------------------------------
Ensures consistent feature ordering per channel.
"""

import pandas as pd
from typing import List


def select_feature_columns(df: pd.DataFrame, channel_name: str) -> List[str]:
    """
    Select all feature columns belonging to a channel.
    Example: Fp1_mean, Fp1_std, Fp1_entropy, ...
    """
    prefix = f"{channel_name}_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols.sort()
    return cols
