"""
==============================================================
WeatherAI
Schema Detection Engine

Author : WeatherAI
Version: 1.0.0

This module is responsible for automatically analysing any
uploaded dataset and producing an intelligent schema profile.

Features
--------
✓ Automatic datatype detection
✓ Timestamp detection
✓ Identifier detection
✓ Missing value analysis
✓ Confidence scoring
✓ Dataset profiling
✓ AI-ready metadata generation

This module is intentionally independent of Streamlit so that
it can be unit tested.
==============================================================
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# -----------------------------------------------------
# Logging
# -----------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

MAX_CATEGORY_RATIO = 0.10
MAX_ID_RATIO = 0.95
BOOLEAN_THRESHOLD = 2
TEXT_AVG_LENGTH = 25

# -----------------------------------------------------
# Common datetime keywords
# -----------------------------------------------------

DATETIME_KEYWORDS = {
    "date",
    "time",
    "timestamp",
    "datetime",
    "created",
    "updated",
    "recorded",
    "observation",
    "event",
    "measurement",
    "logged",
}

# -----------------------------------------------------
# Common target keywords
# -----------------------------------------------------

TARGET_KEYWORDS = {
    "target",
    "class",
    "label",
    "prediction",
    "output",
    "result",
    "temperature",
    "temp",
    "sales",
    "price",
    "score",
    "forecast",
}

# -----------------------------------------------------
# Identifier keywords
# -----------------------------------------------------

ID_KEYWORDS = {
    "id",
    "uuid",
    "identifier",
    "customer",
    "station",
    "device",
    "sensor",
    "employee",
    "record",
    "sample",
}

# -----------------------------------------------------
# Dataclasses
# -----------------------------------------------------

@dataclass
class ColumnProfile:
    """
    Represents the analysis of a single dataframe column.
    """

    name: str

    dtype: str

    inferred_type: str = "unknown"

    missing_count: int = 0

    missing_percentage: float = 0.0

    unique_count: int = 0

    unique_percentage: float = 0.0

    confidence: float = 0.0

    sample_values: List[Any] = field(default_factory=list)

    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """
    Represents the analysis of an entire dataset.
    """

    rows: int = 0

    columns: int = 0

    memory_usage_mb: float = 0.0

    duplicate_rows: int = 0

    duplicate_columns: List[str] = field(default_factory=list)

    numeric_columns: List[str] = field(default_factory=list)

    categorical_columns: List[str] = field(default_factory=list)

    datetime_columns: List[str] = field(default_factory=list)

    boolean_columns: List[str] = field(default_factory=list)

    text_columns: List[str] = field(default_factory=list)

    identifier_columns: List[str] = field(default_factory=list)

    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)


# -----------------------------------------------------
# Main Schema Detector
# -----------------------------------------------------

class SchemaDetector:
    """
    Automatic dataset schema detection.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    DatasetProfile
    """

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

        self.profile = DatasetProfile()

        logger.info("SchemaDetector initialised.")

    # -------------------------------------------------

    def profile_dataset(self) -> DatasetProfile:
        """
        Main entry point.

        Returns
        -------
        DatasetProfile
        """

        logger.info("Profiling dataset...")

        self.profile.rows = len(self.df)

        self.profile.columns = len(self.df.columns)

        self.profile.memory_usage_mb = (
            self.df.memory_usage(deep=True).sum()
            / (1024 ** 2)
        )

        self.profile.duplicate_rows = (
            int(self.df.duplicated().sum())
        )

        self.profile.duplicate_columns = (
            self._find_duplicate_columns()
        )

        for column in self.df.columns:

            profile = self._profile_column(column)

            self.profile.column_profiles[column] = profile

        return self.profile

    # -------------------------------------------------

    def _profile_column(
        self,
        column: str
    ) -> ColumnProfile:
        """
        Produce a basic statistical profile
        for a dataframe column.
        """

        series = self.df[column]

        profile = ColumnProfile(

            name=column,

            dtype=str(series.dtype)

        )

        profile.missing_count = int(series.isna().sum())

        profile.missing_percentage = (
            profile.missing_count
            / max(len(series), 1)
        ) * 100

        profile.unique_count = int(series.nunique(dropna=True))

        profile.unique_percentage = (
            profile.unique_count
            / max(len(series), 1)
        ) * 100

        profile.sample_values = (
            series.dropna()
            .head(5)
            .tolist()
        )

        if pd.api.types.is_numeric_dtype(series):

            profile.statistics = {

                "min": float(series.min()),

                "max": float(series.max()),

                "mean": float(series.mean()),

                "median": float(series.median()),

                "std": float(series.std()),

            }

        return profile

    # -------------------------------------------------

    def _find_duplicate_columns(self):

        """
        Detect duplicated columns.
        """

        duplicates = []

        cols = self.df.columns

        for i in range(len(cols)):

            for j in range(i + 1, len(cols)):

                if self.df.iloc[:, i].equals(
                        self.df.iloc[:, j]
                ):
                    duplicates.append(cols[j])

        return duplicates
    # -------------------------------------------------
    # Intelligent Column Type Detection
    # -------------------------------------------------

    def detect_column_types(self) -> DatasetProfile:
        """
        Analyse every column and infer its semantic type.

        Returns
        -------
        DatasetProfile
        """

        logger.info("Detecting column types...")

        for column in self.df.columns:

            profile = self.profile.column_profiles[column]

            inferred_type, confidence = self._infer_column_type(column)

            profile.inferred_type = inferred_type
            profile.confidence = confidence

            if inferred_type == "numeric":
                self.profile.numeric_columns.append(column)

            elif inferred_type == "categorical":
                self.profile.categorical_columns.append(column)

            elif inferred_type == "datetime":
                self.profile.datetime_columns.append(column)

            elif inferred_type == "boolean":
                self.profile.boolean_columns.append(column)

            elif inferred_type == "text":
                self.profile.text_columns.append(column)

            elif inferred_type == "identifier":
                self.profile.identifier_columns.append(column)

        return self.profile

    # -------------------------------------------------

    def _infer_column_type(
        self,
        column: str
    ) -> tuple[str, float]:
        """
        Infer the semantic type of a column.

        Returns
        -------
        (type, confidence)
        """

        series = self.df[column]

        if self._is_datetime(column):
            return "datetime", 0.99

        if self._is_identifier(column):
            return "identifier", 0.98

        if self._is_boolean(series):
            return "boolean", 0.98

        if self._is_numeric(series):
            return "numeric", 0.95

        if self._is_text(series):
            return "text", 0.90

        return "categorical", 0.85

    # -------------------------------------------------

    def _is_numeric(
        self,
        series: pd.Series
    ) -> bool:
        """
        Determine whether a column is numeric.
        """

        return pd.api.types.is_numeric_dtype(series)

    # -------------------------------------------------

    def _is_boolean(
        self,
        series: pd.Series
    ) -> bool:
        """
        Detect boolean-like columns.
        """

        if pd.api.types.is_bool_dtype(series):
            return True

        values = (
            series.dropna()
            .astype(str)
            .str.lower()
            .unique()
        )

        boolean_values = {
            "0",
            "1",
            "true",
            "false",
            "yes",
            "no",
            "y",
            "n",
            "t",
            "f"
        }

        return (
            len(values) <= BOOLEAN_THRESHOLD
            and set(values).issubset(boolean_values)
        )

    # -------------------------------------------------

    def _is_text(
        self,
        series: pd.Series
    ) -> bool:
        """
        Determine whether a column contains free-form text.
        """

        if not pd.api.types.is_object_dtype(series):
            return False

        clean = series.dropna().astype(str)

        if clean.empty:
            return False

        avg_length = clean.str.len().mean()

        unique_ratio = (
            clean.nunique()
            / max(len(clean), 1)
        )

        return (
            avg_length > TEXT_AVG_LENGTH
            and unique_ratio > 0.50
        )

    # -------------------------------------------------

    def _is_identifier(
        self,
        column: str
    ) -> bool:
        """
        Detect identifier columns.
        """

        series = self.df[column]

        name = column.lower()

        if any(
            keyword in name
            for keyword in ID_KEYWORDS
        ):
            return True

        uniqueness = (
            series.nunique(dropna=True)
            / max(len(series), 1)
        )

        if uniqueness >= MAX_ID_RATIO:
            return True

        return False

    # -------------------------------------------------

    def _is_datetime(
        self,
        column: str
    ) -> bool:
        """
        Detect datetime columns using
        both the column name and contents.
        """

        series = self.df[column]

        column_name = column.lower()

        if any(
            keyword in column_name
            for keyword in DATETIME_KEYWORDS
        ):
            return True

        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        sample = series.dropna().head(100)

        if sample.empty:
            return False

        try:

            converted = pd.to_datetime(
                sample,
                errors="coerce"
            )

            success_rate = (
                converted.notna().mean()
            )

            return success_rate >= 0.80

        except Exception:

            return False

    # -------------------------------------------------

    def get_schema_summary(self) -> Dict[str, Any]:
        """
        Return a lightweight schema summary.
        """

        return {

            "rows": self.profile.rows,

            "columns": self.profile.columns,

            "numeric": self.profile.numeric_columns,

            "categorical": self.profile.categorical_columns,

            "datetime": self.profile.datetime_columns,

            "boolean": self.profile.boolean_columns,

            "text": self.profile.text_columns,

            "identifiers": self.profile.identifier_columns,

            "duplicate_rows": self.profile.duplicate_rows,

            "duplicate_columns": self.profile.duplicate_columns,

            "memory_mb": round(
                self.profile.memory_usage_mb,
                2
            )
        }
    # -------------------------------------------------
    # Time-Series Intelligence
    # -------------------------------------------------

    def detect_timestamp_column(self) -> Optional[str]:
        """
        Detect the most likely timestamp column using
        confidence scoring.

        Returns
        -------
        Optional[str]
            Name of the detected timestamp column or None.
        """

        logger.info("Searching for timestamp column...")

        candidates: List[tuple[str, float]] = []

        for column in self.df.columns:

            confidence = self._timestamp_confidence(column)

            if confidence >= 0.50:
                candidates.append((column, confidence))

        if not candidates:

            logger.info("No timestamp column detected.")

            return None

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        best_column = candidates[0][0]

        logger.info(
            "Timestamp detected: %s (%.2f)",
            best_column,
            candidates[0][1]
        )

        return best_column

    # -------------------------------------------------

    def _timestamp_confidence(
        self,
        column: str
    ) -> float:
        """
        Calculate a confidence score that a column
        represents timestamps.
        """

        score = 0.0

        series = self.df[column]

        name = column.lower()

        # ---------------------------------------------
        # Column name hints
        # ---------------------------------------------

        for keyword in DATETIME_KEYWORDS:

            if keyword in name:
                score += 0.40
                break

        # ---------------------------------------------
        # Already datetime dtype
        # ---------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(series):
            score += 0.50

        # ---------------------------------------------
        # Successful parsing
        # ---------------------------------------------

        sample = series.dropna().head(250)

        if len(sample) > 0:

            parsed = pd.to_datetime(
                sample,
                errors="coerce"
            )

            parse_rate = parsed.notna().mean()

            score += parse_rate * 0.40

        return min(score, 1.0)

    # -------------------------------------------------

    def convert_timestamp_column(
        self,
        column: str
    ) -> pd.Series:
        """
        Convert a timestamp column while attempting
        multiple common date formats.
        """

        logger.info(
            "Converting timestamp column: %s",
            column
        )

        series = self.df[column]

        # Native parsing
        parsed = pd.to_datetime(
            series,
            errors="coerce"
        )

        success = parsed.notna().mean()

        if success >= 0.90:
            return parsed

        # European dates
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=True
        )

        success = parsed.notna().mean()

        if success >= 0.90:
            return parsed

        # Unix seconds
        try:

            parsed = pd.to_datetime(
                series,
                unit="s",
                errors="coerce"
            )

            if parsed.notna().mean() >= 0.90:
                return parsed

        except Exception:

            pass

        # Unix milliseconds
        try:

            parsed = pd.to_datetime(
                series,
                unit="ms",
                errors="coerce"
            )

            if parsed.notna().mean() >= 0.90:
                return parsed

        except Exception:

            pass

        # Excel serial dates
        try:

            parsed = pd.to_datetime(
                series,
                unit="D",
                origin="1899-12-30",
                errors="coerce"
            )

            if parsed.notna().mean() >= 0.90:
                return parsed

        except Exception:

            pass

        logger.warning(
            "Timestamp conversion unsuccessful."
        )

        return pd.to_datetime(
            series,
            errors="coerce"
        )

    # -------------------------------------------------

    def detect_time_series(self) -> bool:
        """
        Determine whether the uploaded dataset is
        suitable for time-series modelling.
        """

        timestamp = self.detect_timestamp_column()

        if timestamp is None:
            return False

        converted = self.convert_timestamp_column(
            timestamp
        )

        valid = converted.notna().sum()

        if valid < 10:
            return False

        ordered = converted.is_monotonic_increasing

        return ordered or valid > len(converted) * 0.80

    # -------------------------------------------------

    def infer_sampling_frequency(
        self,
        timestamp_column: str
    ) -> str:
        """
        Infer the sampling frequency of a time-series.

        Returns
        -------
        str
        """

        ts = self.convert_timestamp_column(
            timestamp_column
        ).dropna()

        if len(ts) < 3:
            return "Unknown"

        delta = ts.sort_values().diff().median()

        if pd.isna(delta):
            return "Unknown"

        seconds = delta.total_seconds()

        if seconds < 60:
            return "Seconds"

        if seconds < 3600:
            return "Minutes"

        if seconds < 86400:
            return "Hourly"

        if seconds < 604800:
            return "Daily"

        if seconds < 2678400:
            return "Weekly"

        if seconds < 31536000:
            return "Monthly"

        return "Yearly"

    # -------------------------------------------------

    def recommend_temporal_features(
        self,
        timestamp_column: str
    ) -> List[str]:
        """
        Recommend feature engineering based on the
        inferred sampling frequency.
        """

        frequency = self.infer_sampling_frequency(
            timestamp_column
        )

        recommendations = [
            "Year",
            "Month",
            "Day",
            "DayOfWeek"
        ]

        if frequency in {
            "Hourly",
            "Minutes",
            "Seconds"
        }:

            recommendations.extend([
                "Hour",
                "Minute",
                "Second"
            ])

        recommendations.extend([
            "Quarter",
            "DayOfYear",
            "WeekOfYear",
            "Weekend",
            "Season",
            "Lag Features",
            "Rolling Mean",
            "Rolling Std",
            "Rolling Median"
        ])

        return recommendations

    # -------------------------------------------------

    def temporal_summary(self) -> Dict[str, Any]:
        """
        Produce a complete summary of the dataset's
        temporal characteristics.
        """

        timestamp = self.detect_timestamp_column()

        if timestamp is None:

            return {

                "time_series": False,

                "timestamp_column": None,

                "frequency": None,

                "recommended_features": []

            }

        return {

            "time_series": self.detect_time_series(),

            "timestamp_column": timestamp,

            "frequency": self.infer_sampling_frequency(
                timestamp
            ),

            "recommended_features":
                self.recommend_temporal_features(
                    timestamp
                )

        }
    # -------------------------------------------------
    # Dataset Health
    # -------------------------------------------------

    def calculate_dataset_health(self) -> Dict[str, Any]:
        """
        Calculate an overall dataset health score.

        Returns
        -------
        dict
        """

        logger.info("Calculating dataset health...")

        score = 100.0

        issues = []
        recommendations = []

        # ---------------------------------------------
        # Duplicate Rows
        # ---------------------------------------------

        if self.profile.duplicate_rows > 0:

            duplicates_pct = (
                self.profile.duplicate_rows
                / max(self.profile.rows, 1)
            ) * 100

            score -= min(duplicates_pct, 10)

            issues.append(
                f"{self.profile.duplicate_rows} duplicate rows detected."
            )

            recommendations.append(
                "Remove duplicate rows."
            )

        # ---------------------------------------------
        # Missing Values
        # ---------------------------------------------

        total_missing = self.df.isna().sum().sum()

        if total_missing > 0:

            missing_pct = (
                total_missing
                / max(self.df.size, 1)
            ) * 100

            score -= min(missing_pct, 20)

            issues.append(
                f"{total_missing} missing values detected."
            )

            recommendations.append(
                "Apply automatic imputation."
            )

        # ---------------------------------------------
        # Duplicate Columns
        # ---------------------------------------------

        if self.profile.duplicate_columns:

            score -= len(
                self.profile.duplicate_columns
            ) * 2

            issues.append(
                f"{len(self.profile.duplicate_columns)} duplicated columns."
            )

            recommendations.append(
                "Remove duplicated columns."
            )

        # ---------------------------------------------
        # Constant Columns
        # ---------------------------------------------

        constant_columns = []

        for col in self.df.columns:

            if self.df[col].nunique(dropna=True) <= 1:

                constant_columns.append(col)

        if constant_columns:

            score -= len(constant_columns) * 2

            issues.append(
                f"{len(constant_columns)} constant columns."
            )

            recommendations.append(
                "Drop constant columns."
            )

        # ---------------------------------------------
        # Identifier Columns
        # ---------------------------------------------

        if self.profile.identifier_columns:

            recommendations.append(
                "Exclude identifier columns from modelling."
            )

        score = max(0, round(score, 2))

        return {

            "health_score": score,

            "issues": issues,

            "recommendations": recommendations,

            "constant_columns": constant_columns

        }

    # -------------------------------------------------
    # AI Summary
    # -------------------------------------------------

    def generate_ai_summary(self) -> str:
        """
        Generate a natural-language summary
        of the uploaded dataset.
        """

        temporal = self.temporal_summary()

        health = self.calculate_dataset_health()

        summary = []

        summary.append(
            f"The uploaded dataset contains "
            f"{self.profile.rows:,} rows and "
            f"{self.profile.columns} columns."
        )

        summary.append(
            f"Detected "
            f"{len(self.profile.numeric_columns)} numeric, "
            f"{len(self.profile.categorical_columns)} categorical, "
            f"{len(self.profile.datetime_columns)} datetime, "
            f"{len(self.profile.boolean_columns)} boolean and "
            f"{len(self.profile.text_columns)} text columns."
        )

        if temporal["time_series"]:

            summary.append(
                f"A time-series structure was detected "
                f"using '{temporal['timestamp_column']}'."
            )

            summary.append(
                f"Estimated sampling frequency: "
                f"{temporal['frequency']}."
            )

        else:

            summary.append(
                "No reliable timestamp column was detected."
            )

        summary.append(
            f"Overall dataset health score: "
            f"{health['health_score']}/100."
        )

        if health["recommendations"]:

            summary.append(
                "Recommended actions: "
                + ", ".join(health["recommendations"])
            )

        return "\n".join(summary)

    # -------------------------------------------------
    # Export
    # -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the complete schema profile.
        """

        return {

            "dataset": {

                "rows": self.profile.rows,

                "columns": self.profile.columns,

                "memory_mb": round(
                    self.profile.memory_usage_mb,
                    2
                )

            },

            "schema": self.get_schema_summary(),

            "temporal": self.temporal_summary(),

            "health": self.calculate_dataset_health(),

            "ai_summary": self.generate_ai_summary()

        }

    # -------------------------------------------------

    def to_json(self) -> str:
        """
        Export schema as JSON.
        """

        import json

        return json.dumps(
            self.to_dict(),
            indent=4,
            default=str
        )

    # -------------------------------------------------
    # Master Analysis
    # -------------------------------------------------

    def analyze(self) -> Dict[str, Any]:
        """
        Complete dataset analysis.

        This is the only public method that
        other modules need to call.
        """

        logger.info("=" * 60)
        logger.info("Starting complete schema analysis...")
        logger.info("=" * 60)

        self.profile_dataset()

        self.detect_column_types()

        results = self.to_dict()

        logger.info("=" * 60)
        logger.info("Schema analysis complete.")
        logger.info("=" * 60)

        return results