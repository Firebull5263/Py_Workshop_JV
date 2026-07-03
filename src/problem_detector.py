"""
==============================================================
WeatherAI
Problem Detection Engine

Author : WeatherAI
Version : 1.0.0

Automatically determines the most appropriate machine
learning problem type and training strategy for an uploaded
dataset.

This module integrates with:

    • SchemaDetector
    • TargetDetector

and produces recommendations for:

    • ML problem type
    • Validation strategy
    • XGBoost model
    • Evaluation metrics
    • Explainability
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

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

MIN_TIME_SERIES_ROWS = 30

MIN_CLUSTER_ROWS = 100

ANOMALY_THRESHOLD = 0.02

MULTI_OUTPUT_THRESHOLD = 2

HIGH_CARDINALITY = 50

# ==========================================================
# Dataclasses
# ==========================================================

@dataclass
class ValidationRecommendation:
    """
    Stores the recommended validation strategy.
    """

    method: str

    folds: int

    shuffle: bool

    random_state: int = 42


@dataclass
class ProblemCandidate:
    """
    Stores one possible ML problem.
    """

    name: str

    confidence: float = 0.0

    score: float = 0.0

    reasoning: List[str] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)


@dataclass
class ProblemDetectionResult:
    """
    Final AI recommendation.
    """

    problem_type: str

    supervised: bool

    confidence: float

    recommended_model: str

    validation: ValidationRecommendation

    metrics: List[str]

    explainability: str

    reasoning: List[str] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)


# ==========================================================
# Main Detector
# ==========================================================

class ProblemDetector:
    """
    Intelligent machine learning problem detector.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    schema : Optional[SchemaDetector]

    target_detector : Optional[TargetDetector]
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        schema: Optional[SchemaDetector] = None,
        target_detector: Optional[TargetDetector] = None,
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

        self.schema = schema

        self.target_detector = target_detector

        self.target_result = target_detector.analyze()

        self.target_column = (
            self.target_result.recommended_target
        )

        self.target_series = self.df[
            self.target_column
        ]

        logger.info(
            "ProblemDetector initialised."
        )

    # ------------------------------------------------------

    def analyze(self) -> ProblemDetectionResult:
        """
        Public entry point.
        """

        logger.info(
            "Beginning problem detection..."
        )

        problem = self._detect_problem()

        validation = self._recommend_validation(
            problem
        )

        model = self._recommend_model(
            problem
        )

        metrics = self._recommend_metrics(
            problem
        )

        explainability = self._recommend_explainability(
            problem
        )

        confidence = self._calculate_confidence(
            problem
        )

        result = ProblemDetectionResult(

            problem_type=problem,

            supervised=self._is_supervised(
                problem
            ),

            confidence=confidence,

            recommended_model=model,

            validation=validation,

            metrics=metrics,

            explainability=explainability,

            reasoning=[],

            warnings=[]

        )

        result.reasoning.extend(

            self._build_reasoning(problem)

        )

        result.warnings.extend(

            self._build_warnings(problem)

        )

        logger.info(
            "Problem detection complete."
        )

        return result
    # ------------------------------------------------------
    # Core Problem Detection
    # ------------------------------------------------------

    def _detect_problem(self) -> str:
        """
        Determine the most appropriate machine learning
        problem type.
        """

        if self._is_time_series():
            return "Time-Series Regression"

        if self._is_multi_output():
            return "Multi-Output Regression"

        target = self.target_series

        if pd.api.types.is_numeric_dtype(target):

            unique = target.nunique(dropna=True)

            if unique <= 2:
                return "Binary Classification"

            if unique <= HIGH_CARDINALITY:

                values = target.dropna().unique()

                try:

                    values = np.sort(values)

                    if np.allclose(
                        values,
                        values.astype(int)
                    ):
                        return "Multiclass Classification"

                except Exception:

                    pass

            return "Regression"

        unique = target.nunique(dropna=True)

        if unique == 2:
            return "Binary Classification"

        return "Multiclass Classification"

    # ------------------------------------------------------

    def _is_supervised(
        self,
        problem: str
    ) -> bool:
        """
        Determine whether the detected problem is supervised.
        """

        return problem not in {

            "Clustering",

            "Anomaly Detection"

        }

    # ------------------------------------------------------

    def _is_time_series(self) -> bool:
        """
        Determine whether the dataset should be treated
        as a forecasting problem.
        """

        temporal = self.schema.temporal_summary()

        if not temporal["time_series"]:
            return False

        if self.df.shape[0] < MIN_TIME_SERIES_ROWS:
            return False

        return True

    # ------------------------------------------------------

    def _is_multi_output(self) -> bool:
        """
        Detect whether multiple potential targets exist.
        """

        numeric = self.schema.profile.numeric_columns

        if len(numeric) < MULTI_OUTPUT_THRESHOLD:
            return False

        possible_targets = []

        for column in numeric:

            if column in self.schema.profile.identifier_columns:
                continue

            unique = self.df[column].nunique()

            if unique > 10:
                possible_targets.append(column)

        return len(possible_targets) >= MULTI_OUTPUT_THRESHOLD

    # ------------------------------------------------------

    def _detect_clustering_candidate(self) -> bool:
        """
        Determine whether clustering could be useful.
        """

        if self.df.shape[0] < MIN_CLUSTER_ROWS:
            return False

        numeric = len(
            self.schema.profile.numeric_columns
        )

        categorical = len(
            self.schema.profile.categorical_columns
        )

        if numeric >= 3 and categorical == 0:
            return True

        return False

    # ------------------------------------------------------

    def _detect_anomaly_candidate(self) -> bool:
        """
        Estimate whether anomaly detection
        is appropriate.
        """

        numeric = self.df.select_dtypes(
            include=np.number
        )

        if numeric.empty:
            return False

        outlier_votes = 0

        for column in numeric.columns:

            series = numeric[column].dropna()

            if len(series) < 20:
                continue

            q1 = series.quantile(0.25)

            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            outliers = (

                (series < lower)

                |

                (series > upper)

            ).mean()

            if outliers > ANOMALY_THRESHOLD:

                outlier_votes += 1

        return outlier_votes >= max(
            1,
            len(numeric.columns) // 3
        )

    # ------------------------------------------------------

    def _problem_score(
        self,
        problem: str
    ) -> float:
        """
        Assign an internal confidence score
        to the detected problem.
        """

        scores = {

            "Time-Series Regression": 99,

            "Regression": 97,

            "Binary Classification": 96,

            "Multiclass Classification": 95,

            "Multi-Output Regression": 93,

            "Clustering": 85,

            "Anomaly Detection": 84

        }

        return scores.get(problem, 80)

    # ------------------------------------------------------

    def _calculate_confidence(
        self,
        problem: str
    ) -> float:
        """
        Produce a calibrated confidence score.
        """

        confidence = self._problem_score(problem)

        health = self.schema.calculate_dataset_health()

        confidence *= (

            health["health_score"] / 100

        )

        confidence = max(
            50.0,
            confidence
        )

        confidence = min(
            99.9,
            confidence
        )

        return round(confidence, 2)
    # ------------------------------------------------------
    # Validation Strategy
    # ------------------------------------------------------

    def _recommend_validation(
        self,
        problem: str
    ) -> ValidationRecommendation:
        """
        Recommend the most appropriate validation strategy.
        """

        if problem == "Time-Series Regression":

            return ValidationRecommendation(
                method="TimeSeriesSplit",
                folds=5,
                shuffle=False
            )

        if problem == "Binary Classification":

            return ValidationRecommendation(
                method="StratifiedKFold",
                folds=5,
                shuffle=True
            )

        if problem == "Multiclass Classification":

            return ValidationRecommendation(
                method="StratifiedKFold",
                folds=5,
                shuffle=True
            )

        if problem == "Multi-Output Regression":

            return ValidationRecommendation(
                method="KFold",
                folds=5,
                shuffle=True
            )

        return ValidationRecommendation(
            method="KFold",
            folds=5,
            shuffle=True
        )

    # ------------------------------------------------------
    # Model Recommendation
    # ------------------------------------------------------

    def _recommend_model(
        self,
        problem: str
    ) -> str:
        """
        Recommend the appropriate XGBoost model.
        """

        models = {

            "Regression":
                "XGBRegressor",

            "Time-Series Regression":
                "XGBRegressor",

            "Binary Classification":
                "XGBClassifier",

            "Multiclass Classification":
                "XGBClassifier",

            "Multi-Output Regression":
                "MultiOutputRegressor(XGBRegressor)",

            "Clustering":
                "KMeans",

            "Anomaly Detection":
                "IsolationForest"

        }

        return models.get(
            problem,
            "XGBRegressor"
        )

    # ------------------------------------------------------
    # Metrics Recommendation
    # ------------------------------------------------------

    def _recommend_metrics(
        self,
        problem: str
    ) -> List[str]:
        """
        Recommend evaluation metrics.
        """

        metric_map = {

            "Regression": [

                "RMSE",
                "MAE",
                "R²",
                "MAPE"

            ],

            "Time-Series Regression": [

                "RMSE",
                "MAE",
                "MAPE",
                "SMAPE"

            ],

            "Binary Classification": [

                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC AUC"

            ],

            "Multiclass Classification": [

                "Accuracy",
                "Macro F1",
                "Weighted F1",
                "Precision",
                "Recall"

            ],

            "Multi-Output Regression": [

                "RMSE",
                "MAE",
                "R²"

            ],

            "Clustering": [

                "Silhouette Score",
                "Davies-Bouldin Index"

            ],

            "Anomaly Detection": [

                "Precision",
                "Recall",
                "ROC AUC"

            ]

        }

        return metric_map.get(problem, [])

    # ------------------------------------------------------
    # Explainability Recommendation
    # ------------------------------------------------------

    def _recommend_explainability(
        self,
        problem: str
    ) -> str:
        """
        Recommend an explainability method.
        """

        if "XGB" in self._recommend_model(problem):

            return "SHAP"

        if problem == "Clustering":

            return "Cluster Feature Importance"

        if problem == "Anomaly Detection":

            return "Isolation Path Analysis"

        return "Feature Importance"

    # ------------------------------------------------------
    # Leakage Detection
    # ------------------------------------------------------

    def detect_data_leakage(self) -> List[str]:
        """
        Detect potential sources of data leakage.
        """

        warnings = []

        target = self.target_column

        numeric = self.df.select_dtypes(
            include=np.number
        )

        if (
            target in numeric.columns
            and len(numeric.columns) > 1
        ):

            corr = numeric.corr(
                numeric_only=True
            )[target]

            suspicious = corr[
                corr.abs() > 0.98
            ].drop(target)

            for column in suspicious.index:

                warnings.append(

                    f"'{column}' is almost perfectly "
                    f"correlated with the target."

                )

        duplicates = self.schema.profile.duplicate_columns

        if duplicates:

            warnings.append(

                "Duplicated columns detected."

            )

        identifiers = self.schema.profile.identifier_columns

        if identifiers:

            warnings.append(

                "Identifier columns should not "
                "be used as predictors."

            )

        return warnings

    # ------------------------------------------------------
    # Reasoning Builder
    # ------------------------------------------------------

    def _build_reasoning(
        self,
        problem: str
    ) -> List[str]:
        """
        Generate AI reasoning.
        """

        reasoning = []

        reasoning.append(

            f"Target column detected: "
            f"{self.target_column}"

        )

        reasoning.append(

            f"Detected ML problem: "
            f"{problem}"

        )

        reasoning.append(

            f"Recommended model: "
            f"{self._recommend_model(problem)}"

        )

        reasoning.append(

            f"Validation strategy: "
            f"{self._recommend_validation(problem).method}"

        )

        reasoning.append(

            f"Explainability: "
            f"{self._recommend_explainability(problem)}"

        )

        return reasoning

    # ------------------------------------------------------
    # Warning Builder
    # ------------------------------------------------------

    def _build_warnings(
        self,
        problem: str
    ) -> List[str]:
        """
        Build a list of workflow warnings.
        """

        warnings = []

        warnings.extend(
            self.detect_data_leakage()
        )

        if self.df.shape[0] < 100:

            warnings.append(

                "Small dataset detected. "
                "Model performance may vary."

            )

        if self.df.isna().sum().sum() > 0:

            warnings.append(

                "Missing values detected. "
                "Preprocessing is recommended."

            )

        if (
            problem == "Time-Series Regression"
            and not self.schema.detect_time_series()
        ):

            warnings.append(

                "Time ordering could not be fully verified."

            )

        return warnings
    # ------------------------------------------------------
    # Executive AI Summary
    # ------------------------------------------------------

    def generate_summary(self) -> str:
        """
        Generate a natural-language summary of the detected
        machine learning workflow.
        """

        result = self.analyze()

        lines = []

        lines.append("AI Workflow Summary")
        lines.append("=" * 40)

        lines.append(
            f"The uploaded dataset is best treated as a "
            f"{result.problem_type} problem."
        )

        lines.append(
            f"The recommended prediction target is "
            f"'{self.target_column}'."
        )

        lines.append(
            f"The suggested model is "
            f"{result.recommended_model}."
        )

        lines.append(
            f"The recommended validation strategy is "
            f"{result.validation.method} "
            f"using {result.validation.folds} folds."
        )

        lines.append(
            f"The preferred explainability framework is "
            f"{result.explainability}."
        )

        lines.append(
            f"Overall confidence: "
            f"{result.confidence:.1f}%."
        )

        if result.reasoning:

            lines.append("")
            lines.append("Reasoning:")

            for item in result.reasoning:

                lines.append(f"• {item}")

        if result.warnings:

            lines.append("")
            lines.append("Warnings:")

            for warning in result.warnings:

                lines.append(f"⚠ {warning}")

        return "\n".join(lines)

    # ------------------------------------------------------
    # Streamlit Summary Card
    # ------------------------------------------------------

    def summary_card(self) -> Dict[str, Any]:
        """
        Lightweight summary for Streamlit dashboards.
        """

        result = self.analyze()

        return {

            "problem_type": result.problem_type,

            "confidence": result.confidence,

            "recommended_model":
                result.recommended_model,

            "validation":
                result.validation.method,

            "metrics":
                result.metrics,

            "explainability":
                result.explainability

        }

    # ------------------------------------------------------
    # Dictionary Export
    # ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the complete recommendation.
        """

        result = self.analyze()

        return {

            "problem_type":
                result.problem_type,

            "supervised":
                result.supervised,

            "confidence":
                result.confidence,

            "target":
                self.target_column,

            "recommended_model":
                result.recommended_model,

            "validation": {

                "method":
                    result.validation.method,

                "folds":
                    result.validation.folds,

                "shuffle":
                    result.validation.shuffle

            },

            "metrics":
                result.metrics,

            "explainability":
                result.explainability,

            "reasoning":
                result.reasoning,

            "warnings":
                result.warnings

        }

    # ------------------------------------------------------

    def to_json(self) -> str:
        """
        Export the recommendation as JSON.
        """

        import json

        return json.dumps(

            self.to_dict(),

            indent=4,

            default=str

        )

    # ------------------------------------------------------
    # Console Report
    # ------------------------------------------------------

    def print_report(self) -> None:
        """
        Print a formatted report for development
        and debugging.
        """

        result = self.analyze()

        print("=" * 70)
        print("WEATHERAI - PROBLEM DETECTOR")
        print("=" * 70)

        print()

        print(f"Target               : {self.target_column}")

        print(f"Problem Type         : {result.problem_type}")

        print(f"Supervised           : {result.supervised}")

        print(f"Confidence           : {result.confidence:.1f}%")

        print(f"Recommended Model    : {result.recommended_model}")

        print(
            f"Validation Strategy  : "
            f"{result.validation.method}"
        )

        print(
            f"Explainability       : "
            f"{result.explainability}"
        )

        print()

        print("Evaluation Metrics")

        for metric in result.metrics:

            print(f"  • {metric}")

        print()

        print("Reasoning")

        for reason in result.reasoning:

            print(f"  • {reason}")

        print()

        if result.warnings:

            print("Warnings")

            for warning in result.warnings:

                print(f"  ⚠ {warning}")

        print("=" * 70)

    # ------------------------------------------------------
    # Health Assessment
    # ------------------------------------------------------

    def workflow_health(self) -> Dict[str, Any]:
        """
        Assess readiness for model training.
        """

        dataset_health = self.schema.calculate_dataset_health()

        leakage = self.detect_data_leakage()

        readiness = "Ready"

        if dataset_health["health_score"] < 70:

            readiness = "Needs Cleaning"

        if leakage:

            readiness = "Review Recommended"

        return {

            "status": readiness,

            "dataset_health":
                dataset_health["health_score"],

            "leakage_count":
                len(leakage),

            "warnings":
                leakage

        }

    # ------------------------------------------------------
    # Master Pipeline Decision
    # ------------------------------------------------------

    def pipeline_configuration(self) -> Dict[str, Any]:
        """
        Produce a single configuration object
        that downstream modules can consume.
        """

        result = self.analyze()

        health = self.workflow_health()

        return {

            "target":

                self.target_column,

            "problem":

                result.problem_type,

            "model":

                result.recommended_model,

            "validation":

                result.validation.method,

            "metrics":

                result.metrics,

            "explainability":

                result.explainability,

            "confidence":

                result.confidence,

            "workflow_health":

                health,

            "time_series":

                self.schema.detect_time_series(),

            "timestamp":

                self.schema.detect_timestamp_column()

        }