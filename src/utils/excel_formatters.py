# src/utils/excel_formatters.py
import pandas as pd
import re

ILLEGAL_CTRL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

def sanitize_scalar_for_excel(x):
    """Strip control chars and ensure safety for Excel export."""
    if isinstance(x, (int, float, pd.Timestamp)): 
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    return ILLEGAL_CTRL_RE.sub("", str(x))

def sanitize_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Applies sanitization to an entire DataFrame."""
    return df.map(sanitize_scalar_for_excel)