"""
excel_formatters.py
===================
Sanitises DataFrames for safe Excel export.

Functions:
    sanitize_scalar_for_excel(x) → any
        Strips illegal control characters from string values so openpyxl
        does not raise IllegalCharacterError on write.
    sanitize_df_for_excel(df) → DataFrame
        Applies sanitize_scalar_for_excel to every cell in a DataFrame.

Input:  pandas DataFrame or scalar value (potentially containing control chars)
Output: sanitised DataFrame / scalar ready to pass to df.to_excel()
"""

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