"""
Shared utility functions.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


TIMESTAMP_KEYWORDS = (
    "time",
    "timestamp",
    "datetime",
    "date",
    "created",
    "recorded",
)

TARGET_KEYWORDS = (
    "target",
    "label",
    "class",
    "output",
    "prediction",
    "temperature",
    "sales",
    "price",
)


def guess_timestamp(df: pd.DataFrame) -> Optional[str]:
    """
    Attempt to identify the timestamp column.
    """

    for col in df.columns:

        if any(
            keyword in col.lower()
            for keyword in TIMESTAMP_KEYWORDS
        ):
            return col

    for col in df.columns:

        try:

            converted = pd.to_datetime(df[col])

            if converted.notna().mean() > 0.90:
                return col

        except Exception:
            pass

    return None


def guess_target(df: pd.DataFrame) -> Optional[str]:
    """
    Attempt to identify the target column.
    """

    for col in df.columns:

        if any(
            keyword in col.lower()
            for keyword in TARGET_KEYWORDS
        ):
            return col

    return df.columns[-1]


def is_identifier(series: pd.Series) -> bool:
    """
    Detect identifier columns.
    """

    return (
        series.nunique(dropna=False)
        >= len(series) * 0.95
    )
