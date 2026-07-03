"""
==============================================================
WeatherAI
AI Target Detection Engine

Author : WeatherAI
Version : 1.0.0

Automatically determines the most likely prediction target
for any uploaded dataset.

The detector assigns confidence scores to every column and
explains WHY a column was selected.

Designed to integrate with SchemaDetector.
==============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.schema_detector import SchemaDetector

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

MAX_NAME_SCORE = 25
MAX_DATATYPE_SCORE = 20
MAX_CARDINALITY_SCORE = 15
MAX_MISSING_SCORE = 10
MAX_VARIANCE_SCORE = 10
MAX_CORRELATION_SCORE = 10

IDENTIFIER_PENALTY = -25
DATETIME_PENALTY = -30
CONSTANT_COLUMN_PENALTY = -20

# ------------------------------------------------------------
# Common target keywords
# ------------------------------------------------------------

TARGET_KEYWORDS = {

    "target",
    "label",
    "class",
    "output",
    "prediction",
    "result",
    "temperature",
    "temp",
    "humidity",
    "pressure",
    "sales",
    "price",
    "forecast",
    "rainfall",
    "precipitation",
    "amount",
    "score",
    "risk",
    "status"

}

# ------------------------------------------------------------
# Binary keywords
# ------------------------------------------------------------

BINARY_KEYWORDS = {

    "yes",
    "no",
    "true",
    "false",
    "fraud",
    "spam",
    "approved",
    "accepted",
    "purchased",
    "default",
    "success"

}

# ============================================================
# Dataclasses
# ============================================================

@dataclass
class TargetCandidate:
    """
    Represents one possible prediction target.
    """

    column: str

    confidence: float = 0.0

    total_score: float = 0.0

    detected_problem: str = "Unknown"

    reasoning: List[str] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)

    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetDetectionResult:
    """
    Final recommendation.
    """

    recommended_target: Optional[str] = None

    confidence: float = 0.0

    detected_problem: str = "Unknown"

    explanation: str = ""

    ranking: List[TargetCandidate] = field(default_factory=list)


# ============================================================
# Main Detector
# ============================================================

class TargetDetector:
    """
    Intelligent AI target detector.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    schema : Optional[SchemaDetector]
    """

    def __init__(

        self,

        dataframe: pd.DataFrame,

        schema: Optional[SchemaDetector] = None

    ):

        self.df = dataframe.copy()

        if schema is None:

            schema = SchemaDetector(self.df)

            schema.analyze()

        self.schema = schema

        self.candidates: List[TargetCandidate] = []

        logger.info("TargetDetector initialised.")

    # --------------------------------------------------------

    def analyze(self) -> TargetDetectionResult:
        """
        Main public entry point.

        Returns
        -------
        TargetDetectionResult
        """

        logger.info(
            "Beginning target analysis..."
        )

        self.candidates.clear()

        for column in self.df.columns:

            candidate = self._evaluate_column(column)

            self.candidates.append(candidate)

        self.candidates.sort(

            key=lambda c: c.total_score,

            reverse=True

        )

        best = self.candidates[0]

        result = TargetDetectionResult(

            recommended_target=best.column,

            confidence=min(
                best.total_score,
                100
            ),

            detected_problem=best.detected_problem,

            explanation=self._generate_explanation(best),

            ranking=self.candidates

        )

        logger.info(
            "Target detection complete."
        )

        return result
    # --------------------------------------------------------
    # Candidate Evaluation
    # --------------------------------------------------------

    def _evaluate_column(
        self,
        column: str
    ) -> TargetCandidate:
        """
        Evaluate a single column as a potential prediction target.
        """

        series = self.df[column]

        candidate = TargetCandidate(column=column)

        score = 0.0

        # ---------------------------------------------
        # Name Score
        # ---------------------------------------------

        name_score = self._score_name(column)
        score += name_score

        if name_score > 0:
            candidate.reasoning.append(
                f"Column name suggests a prediction target (+{name_score:.1f})"
            )

        # ---------------------------------------------
        # Datatype Score
        # ---------------------------------------------

        datatype_score = self._score_datatype(series)
        score += datatype_score

        candidate.reasoning.append(
            f"Datatype score: +{datatype_score:.1f}"
        )

        # ---------------------------------------------
        # Missing Values
        # ---------------------------------------------

        missing_score = self._score_missing(series)
        score += missing_score

        candidate.reasoning.append(
            f"Missing value score: +{missing_score:.1f}"
        )

        # ---------------------------------------------
        # Cardinality
        # ---------------------------------------------

        cardinality_score = self._score_cardinality(series)
        score += cardinality_score

        candidate.reasoning.append(
            f"Cardinality score: +{cardinality_score:.1f}"
        )

        # ---------------------------------------------
        # Variance
        # ---------------------------------------------

        variance_score = self._score_variance(series)
        score += variance_score

        candidate.reasoning.append(
            f"Variance score: +{variance_score:.1f}"
        )

        # ---------------------------------------------
        # Penalties
        # ---------------------------------------------

        penalties = self._calculate_penalties(column)

        score += penalties

        if penalties < 0:

            candidate.warnings.append(
                f"Penalty applied ({penalties:.1f})"
            )

candidate.total_score = round(score, 2)

candidate.metrics = {

    "name_score": name_score,

    "datatype_score": datatype_score,

    "missing_score": missing_score,

    "cardinality_score": cardinality_score,

    "variance_score": variance_score,

    "penalty": penalties

}

return self._finalise_candidate(candidate)

    # --------------------------------------------------------
    # Individual Scoring Functions
    # --------------------------------------------------------

    def _score_name(
        self,
        column: str
    ) -> float:
        """
        Score based on the column name.
        """

        column_name = column.lower()

        score = 0.0

        for keyword in TARGET_KEYWORDS:

            if keyword in column_name:

                score = MAX_NAME_SCORE

                break

        return score

    # --------------------------------------------------------

    def _score_datatype(
        self,
        series: pd.Series
    ) -> float:
        """
        Score based on datatype suitability.
        """

        if pd.api.types.is_numeric_dtype(series):

            return MAX_DATATYPE_SCORE

        if pd.api.types.is_bool_dtype(series):

            return MAX_DATATYPE_SCORE * 0.80

        if pd.api.types.is_object_dtype(series):

            unique = series.nunique(dropna=True)

            if unique <= 20:

                return MAX_DATATYPE_SCORE * 0.70

        return MAX_DATATYPE_SCORE * 0.20

    # --------------------------------------------------------

    def _score_missing(
        self,
        series: pd.Series
    ) -> float:
        """
        Penalise columns with many missing values.
        """

        pct_missing = (
            series.isna().mean()
        )

        score = MAX_MISSING_SCORE * (1 - pct_missing)

        return round(score, 2)

    # --------------------------------------------------------

    def _score_cardinality(
        self,
        series: pd.Series
    ) -> float:
        """
        Score based on uniqueness ratio.
        """

        unique_ratio = (

            series.nunique(dropna=True)

            /

            max(len(series), 1)

        )

        if unique_ratio < 0.02:

            return MAX_CARDINALITY_SCORE * 0.30

        if unique_ratio < 0.20:

            return MAX_CARDINALITY_SCORE

        if unique_ratio < 0.80:

            return MAX_CARDINALITY_SCORE * 0.75

        return MAX_CARDINALITY_SCORE * 0.20

    # --------------------------------------------------------

    def _score_variance(
        self,
        series: pd.Series
    ) -> float:
        """
        Score numeric variance.
        """

        if not pd.api.types.is_numeric_dtype(series):

            return MAX_VARIANCE_SCORE * 0.30

        if series.nunique(dropna=True) <= 1:

            return 0.0

        variance = series.var()

        if pd.isna(variance):

            return 0.0

        if variance <= 0:

            return 0.0

        return MAX_VARIANCE_SCORE

    # --------------------------------------------------------

    def _calculate_penalties(
        self,
        column: str
    ) -> float:
        """
        Apply penalties to unsuitable target columns.
        """

        penalty = 0.0

        profile = self.schema.profile

        if column in profile.identifier_columns:

            penalty += IDENTIFIER_PENALTY

        if column in profile.datetime_columns:

            penalty += DATETIME_PENALTY

        if self.df[column].nunique(dropna=True) <= 1:

            penalty += CONSTANT_COLUMN_PENALTY

        return penalty
    # --------------------------------------------------------
    # Problem Type Detection
    # --------------------------------------------------------

    def _detect_problem_type(
        self,
        series: pd.Series
    ) -> str:
        """
        Automatically determine the machine learning problem type.
        """

        if pd.api.types.is_numeric_dtype(series):

            unique = series.nunique(dropna=True)

            if unique <= 2:
                return "Binary Classification"

            if unique <= 20:

                values = np.sort(series.dropna().unique())

                if np.allclose(values, values.astype(int)):
                    return "Multiclass Classification"

            return "Regression"

        unique = series.nunique(dropna=True)

        if unique == 2:
            return "Binary Classification"

        return "Multiclass Classification"

    # --------------------------------------------------------

    def _score_problem_type(
        self,
        series: pd.Series
    ) -> float:
        """
        Reward columns that are suitable ML targets.
        """

        problem = self._detect_problem_type(series)

        if problem == "Regression":
            return 10.0

        if problem == "Binary Classification":
            return 9.0

        if problem == "Multiclass Classification":
            return 8.0

        return 0.0

    # --------------------------------------------------------
    # Correlation Score
    # --------------------------------------------------------

    def _score_correlation(
        self,
        column: str
    ) -> float:
        """
        Estimate how informative a target is by
        measuring its correlation with the remaining
        numeric features.
        """

        series = self.df[column]

        if not pd.api.types.is_numeric_dtype(series):
            return MAX_CORRELATION_SCORE * 0.5

        numeric = self.df.select_dtypes(
            include=np.number
        )

        if column not in numeric.columns:
            return 0.0

        correlations = numeric.corr(
            numeric_only=True
        )[column].drop(column)

        if correlations.empty:
            return 0.0

        mean_corr = correlations.abs().mean()

        return round(

            min(
                mean_corr * MAX_CORRELATION_SCORE * 2,
                MAX_CORRELATION_SCORE
            ),

            2

        )

    # --------------------------------------------------------
    # Class Balance
    # --------------------------------------------------------

    def _score_class_balance(
        self,
        series: pd.Series
    ) -> float:
        """
        Reward balanced classification targets.
        """

        problem = self._detect_problem_type(series)

        if problem == "Regression":
            return 0.0

        proportions = (

            series.value_counts(
                normalize=True,
                dropna=True
            )

        )

        if proportions.empty:
            return 0.0

        imbalance = proportions.max()

        if imbalance <= 0.60:
            return 10.0

        if imbalance <= 0.75:
            return 8.0

        if imbalance <= 0.90:
            return 5.0

        return 2.0

    # --------------------------------------------------------
    # Confidence Calibration
    # --------------------------------------------------------

    def _calculate_confidence(
        self,
        score: float
    ) -> float:
        """
        Convert a raw score into a confidence value.
        """

        confidence = score

        confidence = max(0.0, confidence)

        confidence = min(confidence, 100.0)

        return round(confidence, 2)

    # --------------------------------------------------------
    # Enhanced Evaluation
    # --------------------------------------------------------

    def _finalise_candidate(
        self,
        candidate: TargetCandidate
    ) -> TargetCandidate:
        """
        Add higher-level intelligence to a candidate.
        """

        series = self.df[candidate.column]

        problem = self._detect_problem_type(series)

        candidate.detected_problem = problem

        problem_score = self._score_problem_type(series)

        candidate.total_score += problem_score

        candidate.reasoning.append(

            f"Detected problem: {problem} (+{problem_score:.1f})"

        )

        correlation_score = self._score_correlation(
            candidate.column
        )

        candidate.total_score += correlation_score

        candidate.reasoning.append(

            f"Feature correlation (+{correlation_score:.1f})"

        )

        if problem != "Regression":

            balance_score = self._score_class_balance(
                series
            )

            candidate.total_score += balance_score

            candidate.reasoning.append(

                f"Class balance (+{balance_score:.1f})"

            )

        candidate.confidence = self._calculate_confidence(

            candidate.total_score

        )

        return candidate
    # --------------------------------------------------------
    # Natural Language Explanation
    # --------------------------------------------------------

    def _generate_explanation(
        self,
        candidate: TargetCandidate
    ) -> str:
        """
        Generate a human-readable explanation for why
        this column was selected as the prediction target.
        """

        lines = []

        lines.append(
            f"The recommended prediction target is "
            f"'{candidate.column}'."
        )

        lines.append("")

        lines.append(
            f"Detected problem type: "
            f"{candidate.detected_problem}."
        )

        lines.append(
            f"Overall confidence: "
            f"{candidate.confidence:.1f}%."
        )

        lines.append("")

        if candidate.reasoning:

            lines.append("Reasons:")

            for reason in candidate.reasoning:

                lines.append(f"• {reason}")

        if candidate.warnings:

            lines.append("")

            lines.append("Warnings:")

            for warning in candidate.warnings:

                lines.append(f"• {warning}")

        return "\n".join(lines)

    # --------------------------------------------------------
    # Candidate Ranking
    # --------------------------------------------------------

    def get_ranked_candidates(self) -> pd.DataFrame:
        """
        Return a ranked DataFrame of all candidate targets.
        """

        rows = []

        for candidate in self.candidates:

            rows.append({

                "Column": candidate.column,

                "Problem Type": candidate.detected_problem,

                "Confidence": round(
                    candidate.confidence,
                    2
                ),

                "Score": round(
                    candidate.total_score,
                    2
                ),

                "Warnings": "; ".join(
                    candidate.warnings
                )

            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # Recommendation Summary
    # --------------------------------------------------------

    def recommendation_summary(
        self
    ) -> Dict[str, Any]:
        """
        Lightweight summary for Streamlit cards.
        """

        result = self.analyze()

        return {

            "recommended_target":
                result.recommended_target,

            "problem_type":
                result.detected_problem,

            "confidence":
                result.confidence,

            "alternatives":[

                candidate.column

                for candidate in result.ranking[1:6]

            ]

        }

    # --------------------------------------------------------
    # User Override
    # --------------------------------------------------------

    def override_target(
        self,
        column: str
    ) -> TargetDetectionResult:
        """
        Allow the user to manually choose
        another target while preserving
        the AI recommendation.
        """

        if column not in self.df.columns:

            raise ValueError(
                f"{column} not found."
            )

        candidate = self._evaluate_column(
            column
        )

        candidate = self._finalise_candidate(
            candidate
        )

        return TargetDetectionResult(

            recommended_target=column,

            confidence=candidate.confidence,

            detected_problem=candidate.detected_problem,

            explanation=self._generate_explanation(
                candidate
            ),

            ranking=self.candidates

        )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the AI recommendation.
        """

        result = self.analyze()

        return {

            "recommended_target":

                result.recommended_target,

            "confidence":

                result.confidence,

            "problem_type":

                result.detected_problem,

            "explanation":

                result.explanation,

            "ranking":[

                {

                    "column": c.column,

                    "confidence": c.confidence,

                    "score": c.total_score,

                    "problem":

                        c.detected_problem

                }

                for c in result.ranking

            ]

        }

    # --------------------------------------------------------

    def to_json(self) -> str:
        """
        Export recommendation as JSON.
        """

        import json

        return json.dumps(

            self.to_dict(),

            indent=4,

            default=str

        )

    # --------------------------------------------------------

    def print_report(self):
        """
        Print a formatted console report.
        Useful during development.
        """

        result = self.analyze()

        print("=" * 60)

        print("AI TARGET DETECTOR")

        print("=" * 60)

        print()

        print(
            f"Recommended Target : "
            f"{result.recommended_target}"
        )

        print(
            f"Problem Type       : "
            f"{result.detected_problem}"
        )

        print(
            f"Confidence         : "
            f"{result.confidence:.1f}%"
        )

        print()

        print(result.explanation)

        print()

        print("=" * 60)

        print("Top Candidates")

        print("=" * 60)

        print(

            self.get_ranked_candidates()

        )

        print("=" * 60)