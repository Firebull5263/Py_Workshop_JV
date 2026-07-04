"""
==============================================================
WeatherAI
AI Data Quality Engine

Author : WeatherAI
Version : 1.0.0

The Data Quality Engine performs a comprehensive assessment
of uploaded datasets, identifies quality issues, recommends
cleaning actions, and generates an AI-driven cleaning plan.

The engine is intentionally NON-DESTRUCTIVE. It analyses the
dataset first and produces a cleaning plan before applying
any transformations.

Integrates with:

    • SchemaDetector
    • TargetDetector
    • ProblemDetector
==============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.schema_detector import SchemaDetector
from src.target_detector import TargetDetector
from src.problem_detector import ProblemDetector

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

SPARSE_COLUMN_THRESHOLD = 0.80

HIGH_CARDINALITY_RATIO = 0.50

CONSTANT_THRESHOLD = 1

NEAR_CONSTANT_THRESHOLD = 0.98

MAX_MISSING_FOR_DROP = 0.60

DEFAULT_RANDOM_STATE = 42

# ==========================================================
# Dataclasses
# ==========================================================

@dataclass
class DataIssue:
    """
    Represents one detected quality issue.
    """

    issue_type: str

    severity: str

    column: Optional[str] = None

    description: str = ""

    recommendation: str = ""


@dataclass
class CleaningAction:
    """
    Represents one recommended cleaning action.
    """

    action: str

    column: Optional[str] = None

    reason: str = ""

    automatic: bool = True

    priority: int = 1


@dataclass
class DataQualityReport:
    """
    Final report produced by the engine.
    """

    health_score: float

    issues: List[DataIssue] = field(default_factory=list)

    cleaning_plan: List[CleaningAction] = field(default_factory=list)

    statistics: Dict[str, Any] = field(default_factory=dict)

    warnings: List[str] = field(default_factory=list)


# ==========================================================
# Main Engine
# ==========================================================

class DataQualityEngine:
    """
    Intelligent dataset quality assessment.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    schema : Optional[SchemaDetector]

    target_detector : Optional[TargetDetector]

    problem_detector : Optional[ProblemDetector]
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        schema: Optional[SchemaDetector] = None,
        target_detector: Optional[TargetDetector] = None,
        problem_detector: Optional[ProblemDetector] = None,
    ):

        self.df = dataframe.copy()

        if schema is None:

            schema = SchemaDetector(self.df)

            schema.analyze()

        if target_detector is None:

            target_detector = TargetDetector(
                self.df,
                schema=schema
            )

        if problem_detector is None:

            problem_detector = ProblemDetector(
                self.df,
                schema=schema,
                target_detector=target_detector
            )

        self.schema = schema

        self.target_detector = target_detector

        self.problem_detector = problem_detector

        self.issues: List[DataIssue] = []

        self.cleaning_plan: List[CleaningAction] = []

        logger.info(
            "DataQualityEngine initialised."
        )

    # ------------------------------------------------------

    def analyze(self) -> DataQualityReport:
        """
        Public entry point.

        Performs a complete quality assessment without
        modifying the dataset.
        """

        logger.info(
            "Beginning AI data quality assessment..."
        )

        self.issues.clear()

        self.cleaning_plan.clear()

        self._inspect_missing_values()

        self._inspect_duplicates()

        self._inspect_constant_columns()

        self._inspect_sparse_columns()

        self._inspect_high_cardinality()

        self._inspect_invalid_types()

        self._inspect_datetime_columns()

        self._inspect_identifier_columns()
      
        self._inspect_outliers()

        self._inspect_class_imbalance()

        self._inspect_distributions()

        self._inspect_memory_usage()

        report = DataQualityReport(

            health_score=self._calculate_health_score(),

            issues=self.issues,

            cleaning_plan=self.cleaning_plan,

            statistics=self._build_statistics(),

            warnings=self._build_warnings()

        )

        logger.info(
            "Data quality assessment complete."
        )

        return report
    # ------------------------------------------------------
    # Missing Value Inspection
    # ------------------------------------------------------

    def _inspect_missing_values(self) -> None:
        """
        Inspect all columns for missing values and recommend
        appropriate imputation strategies.
        """

        logger.info("Inspecting missing values...")

        total_rows = len(self.df)

        for column in self.df.columns:

            missing = self.df[column].isna().sum()

            if missing == 0:
                continue

            ratio = missing / max(total_rows, 1)

            severity = (
                "High" if ratio >= 0.50
                else "Medium" if ratio >= 0.20
                else "Low"
            )

            self.issues.append(
                DataIssue(
                    issue_type="Missing Values",
                    severity=severity,
                    column=column,
                    description=(
                        f"{missing} missing values "
                        f"({ratio:.1%})"
                    ),
                    recommendation=self._recommend_imputation(column)
                )
            )

            if ratio >= MAX_MISSING_FOR_DROP:

                self.cleaning_plan.append(
                    CleaningAction(
                        action="Drop Column",
                        column=column,
                        reason="Excessive missing values",
                        priority=1
                    )
                )

            else:

                self.cleaning_plan.append(
                    CleaningAction(
                        action=self._recommend_imputation(column),
                        column=column,
                        reason="Fill missing values",
                        priority=2
                    )
                )

    # ------------------------------------------------------
    # Duplicate Inspection
    # ------------------------------------------------------

    def _inspect_duplicates(self) -> None:
        """
        Inspect duplicate rows and duplicate columns.
        """

        logger.info("Inspecting duplicates...")

        duplicate_rows = int(
            self.df.duplicated().sum()
        )

        if duplicate_rows > 0:

            self.issues.append(
                DataIssue(
                    issue_type="Duplicate Rows",
                    severity="Medium",
                    description=f"{duplicate_rows} duplicate rows detected.",
                    recommendation="Remove duplicate rows."
                )
            )

            self.cleaning_plan.append(
                CleaningAction(
                    action="Remove Duplicate Rows",
                    reason="Duplicate observations detected.",
                    priority=1
                )
            )

        duplicates = self.schema.profile.duplicate_columns

        for column in duplicates:

            self.issues.append(
                DataIssue(
                    issue_type="Duplicate Column",
                    severity="Medium",
                    column=column,
                    description="Column duplicates another feature.",
                    recommendation="Remove duplicate column."
                )
            )

            self.cleaning_plan.append(
                CleaningAction(
                    action="Drop Column",
                    column=column,
                    reason="Duplicate feature",
                    priority=1
                )
            )

    # ------------------------------------------------------
    # Constant Columns
    # ------------------------------------------------------

    def _inspect_constant_columns(self) -> None:
        """
        Detect constant and near-constant features.
        """

        logger.info("Inspecting constant columns...")

        for column in self.df.columns:

            series = self.df[column]

            unique = series.nunique(dropna=True)

            if unique <= CONSTANT_THRESHOLD:

                self.issues.append(
                    DataIssue(
                        issue_type="Constant Feature",
                        severity="High",
                        column=column,
                        description="Feature contains only one unique value.",
                        recommendation="Drop feature."
                    )
                )

                self.cleaning_plan.append(
                    CleaningAction(
                        action="Drop Column",
                        column=column,
                        reason="Constant feature",
                        priority=1
                    )
                )

                continue

            frequency = (
                series.value_counts(normalize=True, dropna=True)
            )

            if (
                not frequency.empty
                and frequency.iloc[0] >= NEAR_CONSTANT_THRESHOLD
            ):

                self.issues.append(
                    DataIssue(
                        issue_type="Near Constant Feature",
                        severity="Low",
                        column=column,
                        description="Feature has extremely low variance.",
                        recommendation="Consider removing feature."
                    )
                )

    # ------------------------------------------------------
    # Sparse Columns
    # ------------------------------------------------------

    def _inspect_sparse_columns(self) -> None:
        """
        Detect sparse features.
        """

        logger.info("Inspecting sparse columns...")

        for column in self.df.columns:

            ratio = self.df[column].isna().mean()

            if ratio >= SPARSE_COLUMN_THRESHOLD:

                self.issues.append(
                    DataIssue(
                        issue_type="Sparse Feature",
                        severity="Medium",
                        column=column,
                        description=f"{ratio:.1%} missing values.",
                        recommendation="Review usefulness."
                    )
                )

    # ------------------------------------------------------
    # High Cardinality
    # ------------------------------------------------------

    def _inspect_high_cardinality(self) -> None:
        """
        Detect categorical columns with excessive cardinality.
        """

        logger.info("Inspecting cardinality...")

        rows = max(len(self.df), 1)

        for column in self.schema.profile.categorical_columns:

            unique_ratio = (

                self.df[column].nunique(dropna=True)

                / rows

            )

            if unique_ratio >= HIGH_CARDINALITY_RATIO:

                self.issues.append(
                    DataIssue(
                        issue_type="High Cardinality",
                        severity="Medium",
                        column=column,
                        description=(
                            f"Unique ratio {unique_ratio:.1%}"
                        ),
                        recommendation=(
                            "Consider frequency encoding."
                        )
                    )
                )

    # ------------------------------------------------------
    # Invalid Datatypes
    # ------------------------------------------------------

    def _inspect_invalid_types(self) -> None:
        """
        Detect mixed datatypes in object columns.
        """

        logger.info("Inspecting mixed datatypes...")

        for column in self.df.columns:

            series = self.df[column].dropna()

            if series.empty:
                continue

            detected = {

                type(v).__name__

                for v in series.head(500)

            }

            if len(detected) > 1:

                self.issues.append(
                    DataIssue(
                        issue_type="Mixed Datatypes",
                        severity="Medium",
                        column=column,
                        description=(
                            ", ".join(sorted(detected))
                        ),
                        recommendation="Standardize datatype."
                    )
                )

                self.cleaning_plan.append(
                    CleaningAction(
                        action="Convert Datatype",
                        column=column,
                        reason="Mixed datatypes detected.",
                        priority=2
                    )
                )

    # ------------------------------------------------------
    # Datetime Inspection
    # ------------------------------------------------------

    def _inspect_datetime_columns(self) -> None:
        """
        Validate detected datetime columns.
        """

        logger.info("Inspecting datetime columns...")

        for column in self.schema.profile.datetime_columns:

            parsed = pd.to_datetime(
                self.df[column],
                errors="coerce"
            )

            invalid = parsed.isna().sum()

            if invalid == 0:
                continue

            self.issues.append(
                DataIssue(
                    issue_type="Invalid Datetime",
                    severity="Medium",
                    column=column,
                    description=f"{invalid} invalid dates.",
                    recommendation="Re-parse datetime."
                )
            )

            self.cleaning_plan.append(
                CleaningAction(
                    action="Convert Datetime",
                    column=column,
                    reason="Invalid datetime values.",
                    priority=2
                )
            )

    # ------------------------------------------------------
    # Identifier Inspection
    # ------------------------------------------------------

    def _inspect_identifier_columns(self) -> None:
        """
        Warn about identifier columns.
        """

        logger.info("Inspecting identifier columns...")

        for column in self.schema.profile.identifier_columns:

            self.issues.append(
                DataIssue(
                    issue_type="Identifier",
                    severity="Low",
                    column=column,
                    description="Identifier column detected.",
                    recommendation="Exclude from modelling."
                )
            )
    # ------------------------------------------------------
    # Outlier Inspection
    # ------------------------------------------------------

    def _inspect_outliers(self) -> None:
        """
        Detect outliers in numeric columns using the IQR rule.
        """

        logger.info("Inspecting outliers...")

        numeric_columns = self.schema.profile.numeric_columns

        for column in numeric_columns:

            if column == self.target_detector.analyze().recommended_target:
                continue

            series = self.df[column].dropna()

            if len(series) < 10:
                continue

            q1 = series.quantile(0.25)

            q3 = series.quantile(0.75)

            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            outliers = ((series < lower) | (series > upper))

            count = int(outliers.sum())

            if count == 0:
                continue

            ratio = count / len(series)

            severity = (
                "High"
                if ratio > 0.10
                else "Medium"
                if ratio > 0.03
                else "Low"
            )

            self.issues.append(

                DataIssue(

                    issue_type="Outliers",

                    severity=severity,

                    column=column,

                    description=f"{count} potential outliers detected.",

                    recommendation="Winsorize or Robust Scale"

                )

            )

            self.cleaning_plan.append(

                CleaningAction(

                    action="Treat Outliers",

                    column=column,

                    reason="IQR outlier detection",

                    priority=2

                )

            )

    # ------------------------------------------------------
    # Class Imbalance
    # ------------------------------------------------------

    def _inspect_class_imbalance(self) -> None:
        """
        Evaluate balance of classification targets.
        """

        logger.info("Inspecting class imbalance...")

        result = self.problem_detector.analyze()

        if "Classification" not in result.problem_type:
            return

        target = self.target_detector.analyze().recommended_target

        distribution = (

            self.df[target]

            .value_counts(normalize=True)

        )

        if distribution.empty:
            return

        largest = distribution.iloc[0]

        if largest <= 0.70:
            return

        severity = (

            "High"

            if largest >= 0.90

            else "Medium"

        )

        self.issues.append(

            DataIssue(

                issue_type="Class Imbalance",

                severity=severity,

                column=target,

                description=(
                    f"Largest class occupies "
                    f"{largest:.1%} of observations."
                ),

                recommendation=(
                    "Use class weighting or SMOTE."
                )

            )

        )

    # ------------------------------------------------------
    # Numeric Distribution
    # ------------------------------------------------------

    def _inspect_distributions(self) -> None:
        """
        Identify highly skewed numeric features.
        """

        logger.info("Inspecting feature distributions...")

        numeric = self.schema.profile.numeric_columns

        for column in numeric:

            series = self.df[column].dropna()

            if len(series) < 20:
                continue

            skewness = series.skew()

            if abs(skewness) < 2:
                continue

            self.issues.append(

                DataIssue(

                    issue_type="Highly Skewed",

                    severity="Low",

                    column=column,

                    description=(
                        f"Skewness = {skewness:.2f}"
                    ),

                    recommendation=(
                        "Consider log transformation."
                    )

                )

            )

    # ------------------------------------------------------
    # Memory Optimisation
    # ------------------------------------------------------

    def _inspect_memory_usage(self) -> None:
        """
        Recommend datatype optimisation.
        """

        logger.info("Inspecting memory usage...")

        memory = (

            self.df.memory_usage(deep=True)

            .sum()

            / 1024**2

        )

        if memory < 25:
            return

        self.issues.append(

            DataIssue(

                issue_type="Memory",

                severity="Low",

                description=(
                    f"Dataset uses {memory:.1f} MB."
                ),

                recommendation=(
                    "Downcast numeric datatypes."
                )

            )

        )

        self.cleaning_plan.append(

            CleaningAction(

                action="Optimize Memory",

                reason="Reduce memory footprint.",

                priority=3

            )

        )

    # ------------------------------------------------------
    # Dataset Statistics
    # ------------------------------------------------------

    def _build_statistics(self) -> Dict[str, Any]:
        """
        Build dataset summary statistics.
        """

        return {

            "rows": len(self.df),

            "columns": len(self.df.columns),

            "missing_values": int(
                self.df.isna().sum().sum()
            ),

            "duplicate_rows": int(
                self.df.duplicated().sum()
            ),

            "issues_found": len(self.issues),

            "recommended_actions": len(
                self.cleaning_plan
            ),

            "memory_mb": round(

                self.df.memory_usage(
                    deep=True
                ).sum() / 1024**2,

                2

            )

        }

    # ------------------------------------------------------
    # Dataset Health
    # ------------------------------------------------------

    def _calculate_health_score(self) -> float:
        """
        Calculate an overall quality score.
        """

        score = 100.0

        weights = {

            "High": 8,

            "Medium": 4,

            "Low": 1

        }

        for issue in self.issues:

            score -= weights.get(
                issue.severity,
                2
            )

        score = max(0.0, score)

        return round(score, 2)

    # ------------------------------------------------------
    # Warning Builder
    # ------------------------------------------------------

    def _build_warnings(self) -> List[str]:
        """
        Generate user-friendly warnings.
        """

        warnings = []

        if self._calculate_health_score() < 70:

            warnings.append(

                "Dataset quality is below the recommended threshold."

            )

        if len(self.cleaning_plan) > 10:

            warnings.append(

                "A significant number of cleaning actions are recommended."

            )

        if any(

            issue.issue_type == "Class Imbalance"

            for issue in self.issues

        ):

            warnings.append(

                "Class imbalance may reduce classification performance."

            )

        if any(

            issue.issue_type == "Outliers"

            for issue in self.issues

        ):

            warnings.append(

                "Outliers may negatively influence model accuracy."

            )

        return warnings
    # ------------------------------------------------------
    # Execute Cleaning Plan
    # ------------------------------------------------------

    def apply_cleaning(
        self,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Apply the AI-generated cleaning plan.

        Parameters
        ----------
        inplace : bool
            If True, modify the internal dataframe.
            Otherwise return a cleaned copy.
        """

        logger.info("Applying AI cleaning plan...")

        df = self.df if inplace else self.df.copy()

        audit_log = []

        for action in sorted(
            self.cleaning_plan,
            key=lambda x: x.priority
        ):

            column = action.column

            try:

                # ----------------------------------------
                # Drop Column
                # ----------------------------------------

                if (
                    action.action == "Drop Column"
                    and column in df.columns
                ):

                    df.drop(
                        columns=[column],
                        inplace=True
                    )

                # ----------------------------------------
                # Duplicate Rows
                # ----------------------------------------

                elif action.action == "Remove Duplicate Rows":

                    before = len(df)

                    df.drop_duplicates(
                        inplace=True
                    )

                    after = len(df)

                    audit_log.append(
                        f"Removed {before-after} duplicate rows."
                    )

                # ----------------------------------------
                # Missing Values
                # ----------------------------------------

                elif action.action == "Median Imputation":

                    df[column] = df[column].fillna(
                        df[column].median()
                    )

                elif action.action == "Mean Imputation":

                    df[column] = df[column].fillna(
                        df[column].mean()
                    )

                elif action.action == "Mode Imputation":

                    mode = df[column].mode()

                    if not mode.empty:

                        df[column] = df[column].fillna(
                            mode.iloc[0]
                        )

                # ----------------------------------------
                # Datetime
                # ----------------------------------------

                elif action.action == "Convert Datetime":

                    df[column] = pd.to_datetime(
                        df[column],
                        errors="coerce"
                    )

                # ----------------------------------------
                # Datatype
                # ----------------------------------------

                elif action.action == "Convert Datatype":

                    df[column] = (
                        df[column]
                        .infer_objects(copy=False)
                    )

                # ----------------------------------------
                # Outliers
                # ----------------------------------------

                elif action.action == "Treat Outliers":

                    s = df[column]

                    q1 = s.quantile(0.25)

                    q3 = s.quantile(0.75)

                    iqr = q3 - q1

                    lower = q1 - 1.5 * iqr

                    upper = q3 + 1.5 * iqr

                    df[column] = s.clip(
                        lower,
                        upper
                    )

                # ----------------------------------------
                # Memory
                # ----------------------------------------

                elif action.action == "Optimize Memory":

                    df = self._optimize_memory(df)

                audit_log.append(
                    f"{action.action} ({column})"
                )

            except Exception as exc:

                logger.warning(
                    "Cleaning action failed: %s",
                    exc
                )

                audit_log.append(
                    f"FAILED: {action.action} ({column})"
                )

        self.audit_log = audit_log

        if inplace:
            self.df = df

        logger.info("Cleaning completed.")

        return df

    # ------------------------------------------------------
    # Memory Optimisation
    # ------------------------------------------------------

    def _optimize_memory(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Downcast numeric datatypes.
        """

        df = dataframe.copy()

        for column in df.select_dtypes(
            include=["int"]
        ):

            df[column] = pd.to_numeric(
                df[column],
                downcast="integer"
            )

        for column in df.select_dtypes(
            include=["float"]
        ):

            df[column] = pd.to_numeric(
                df[column],
                downcast="float"
            )

        return df

    # ------------------------------------------------------
    # Cleaning Report
    # ------------------------------------------------------

    def cleaning_report(self) -> Dict[str, Any]:
        """
        Generate a cleaning summary.
        """

        report = self.analyze()

        return {

            "health_score":
                report.health_score,

            "issues":
                len(report.issues),

            "actions":
                len(report.cleaning_plan),

            "warnings":
                report.warnings,

            "audit_log":
                getattr(
                    self,
                    "audit_log",
                    []
                )

        }

    # ------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------

    def executive_summary(self) -> str:
        """
        Generate an AI summary suitable for Streamlit.
        """

        report = self.analyze()

        lines = []

        lines.append(
            "AI Data Quality Assessment"
        )

        lines.append("=" * 40)

        lines.append(
            f"Dataset Health: "
            f"{report.health_score:.1f}/100"
        )

        lines.append("")

        lines.append(
            f"Issues detected: "
            f"{len(report.issues)}"
        )

        lines.append(
            f"Recommended actions: "
            f"{len(report.cleaning_plan)}"
        )

        if report.warnings:

            lines.append("")

            lines.append("Warnings:")

            for warning in report.warnings:

                lines.append(
                    f"⚠ {warning}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------
    # Export
    # ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the quality report.
        """

        report = self.analyze()

        return {

            "health_score":
                report.health_score,

            "statistics":
                report.statistics,

            "issues":[

                issue.__dict__

                for issue in report.issues

            ],

            "cleaning_plan":[

                action.__dict__

                for action in report.cleaning_plan

            ],

            "warnings":
                report.warnings

        }

    # ------------------------------------------------------

    def to_json(self) -> str:
        """
        Export report as JSON.
        """

        import json

        return json.dumps(

            self.to_dict(),

            indent=4,

            default=str

        )

    # ------------------------------------------------------
    # Streamlit Helper
    # ------------------------------------------------------

    def dashboard_summary(self) -> Dict[str, Any]:
        """
        Compact summary for dashboard cards.
        """

        report = self.analyze()

        return {

            "health":
                report.health_score,

            "issues":
                len(report.issues),

            "actions":
                len(report.cleaning_plan),

            "ready":

                report.health_score >= 80

        }

    # ------------------------------------------------------
    # Pipeline Integration
    # ------------------------------------------------------

    def pipeline_configuration(self) -> Dict[str, Any]:
        """
        Return a unified configuration object
        for downstream preprocessing.
        """

        return {

            "quality":
                self.dashboard_summary(),

            "problem":
                self.problem_detector.pipeline_configuration(),

            "target":
                self.target_detector.to_dict(),

            "schema":
                self.schema.to_dict()

        }
