"""
==============================================================
WeatherAI
Adaptive AI Preprocessor

Author : WeatherAI
Version : 1.0.0

Transforms raw datasets into model-ready feature matrices by
combining information from the Schema Detector,
Target Detector, Problem Detector and Data Quality Engine.

The preprocessor automatically adapts itself to unknown
datasets without requiring hard-coded column names.

==============================================================
"""

from __future__ import annotations

import hashlib
import logging
import uuid

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit
)

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer
)

from sklearn.impute import (
    SimpleImputer,
    KNNImputer
)

from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression
)

from src.schema_detector import SchemaDetector
from src.target_detector import TargetDetector
from src.problem_detector import ProblemDetector
from src.data_quality import DataQualityEngine

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

DEFAULT_RANDOM_STATE = 42

DEFAULT_TEST_SIZE = 0.20

HIGH_CARDINALITY = 50

CORRELATION_THRESHOLD = 0.95

LOW_VARIANCE_THRESHOLD = 0.01

# ==========================================================
# Dataclasses
# ==========================================================

@dataclass
class FeatureEngineeringStep:
    """
    Represents one engineered feature.
    """

    feature_name: str

    source_column: str

    transformation: str

    description: str


@dataclass
class PreprocessingReport:
    """
    Human-readable preprocessing report.
    """

    pipeline_version: str

    actions: List[str] = field(default_factory=list)

    engineered_features: List[str] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)

    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingResult:
    """
    Final output returned by the preprocessor.
    """

    cleaned_dataframe: pd.DataFrame

    processed_dataframe: pd.DataFrame

    feature_matrix: Any

    target_vector: Any

    feature_names: List[str]

    pipeline: Optional[Pipeline]

    transformers: Dict[str, Any]

    metadata: Dict[str, Any]

    train_data: Dict[str, Any]

    test_data: Dict[str, Any]

    report: PreprocessingReport

    audit_log: List[str]

# ==========================================================
# Main Engine
# ==========================================================

class AdaptivePreprocessor:
    """
    Enterprise adaptive preprocessing engine.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        schema: Optional[SchemaDetector] = None,
        target_detector: Optional[TargetDetector] = None,
        problem_detector: Optional[ProblemDetector] = None,
        quality_engine: Optional[DataQualityEngine] = None,
    ):

        self.original_df = dataframe.copy()

        self.df = dataframe.copy()

        self.pipeline_version = self._generate_pipeline_version()

        self.audit_log: List[str] = []

        self.feature_steps: List[
            FeatureEngineeringStep
        ] = []

        # ----------------------------------------------

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

        if quality_engine is None:

            quality_engine = DataQualityEngine(
                self.df,
                schema=schema,
                target_detector=target_detector,
                problem_detector=problem_detector
            )

        self.schema = schema

        self.target_detector = target_detector

        self.problem_detector = problem_detector

        self.quality_engine = quality_engine

        self.target = (

            target_detector.analyze()

            .recommended_target

        )

        logger.info(
            "AdaptivePreprocessor initialised."
        )

    # --------------------------------------------------

    def preprocess(self) -> PreprocessingResult:
        """
        Public entry point.
        """

        logger.info(
            "Beginning adaptive preprocessing..."
        )

        # AI Cleaning

        self.df = self.quality_engine.apply_cleaning()

        self.audit_log.append(
            "Applied AI cleaning plan."
        )
        # ----------------------------------
        # Feature Engineering
        # ----------------------------------

        self._feature_engineering()

        # ----------------------------------
        # Build Transformer
        # ----------------------------------

        transformer = self._build_transformer()
        # ----------------------------------
        # Model Preparation
        # ----------------------------------

        (

            X_train,

            X_test,

            y_train,

            y_test,

            X_train_processed,

            X_test_processed,

            feature_names

        ) = self._prepare_model_data(
            transformer
        )

        report = self._report()

        result = PreprocessingResult(

            cleaned_dataframe=self.df.copy(),

            processed_dataframe=self.df.copy(),

            feature_matrix=X_train_processed,

            target_vector=y_train,

            feature_names=feature_names,

            pipeline=transformer,

            transformers={

                "preprocessor": transformer

            },

            metadata=self._metadata(),

            train_data={

                "X": X_train_processed,

                "y": y_train

            },

            test_data={

                "X": X_test_processed,

                "y": y_test

            },

            report=report,

            audit_log=self.audit_log.copy()

        )

        bundle = self.build_bundle(result)

        result.metadata["pipeline_bundle"] = bundle

        return result
      
        report = PreprocessingReport(

            pipeline_version=self.pipeline_version,

            actions=self.audit_log,

            engineered_features=[],

            warnings=[],

            statistics={}

        )

        return PreprocessingResult(

            cleaned_dataframe=self.df.copy(),

            processed_dataframe=self.df.copy(),

            feature_matrix=None,

            target_vector=None,

            feature_names=[],

            pipeline=transformer,

            transformers={
                "preprocessor": transformer
            },

            metadata=self._metadata(),

            train_data={},

            test_data={},

            report=report,

            audit_log=self.audit_log

        )

    # --------------------------------------------------

    def _generate_pipeline_version(self) -> str:
        """
        Generate a reproducible pipeline version.
        """

        fingerprint = hashlib.md5(

            (
                str(uuid.uuid4()) +
                str(self.df.shape)
            ).encode()

        ).hexdigest()[:8]

        return f"2026.07.{fingerprint}"

    # --------------------------------------------------

    def _metadata(self) -> Dict[str, Any]:
        """
        Pipeline metadata.
        """

        return {

            "pipeline_version":
                self.pipeline_version,

            "rows":
                len(self.df),

            "columns":
                len(self.df.columns),

            "target":
                self.target,

            "problem":
                self.problem_detector
                .analyze()
                .problem_type

        }
    # --------------------------------------------------
    # Datetime Feature Engineering
    # --------------------------------------------------

    def _engineer_datetime_features(self) -> None:
        """
        Automatically engineer useful features from
        detected datetime columns.
        """

        logger.info("Engineering datetime features...")

        for column in self.schema.profile.datetime_columns:

            if column not in self.df.columns:
                continue

            dt = pd.to_datetime(
                self.df[column],
                errors="coerce"
            )

            features = {

                f"{column}_year": dt.dt.year,

                f"{column}_month": dt.dt.month,

                f"{column}_day": dt.dt.day,

                f"{column}_weekday": dt.dt.weekday,

                f"{column}_quarter": dt.dt.quarter,

                f"{column}_dayofyear": dt.dt.dayofyear,

                f"{column}_is_weekend": (
                    dt.dt.weekday >= 5
                ).astype(int)

            }

            if not dt.dt.hour.isna().all():

                features[f"{column}_hour"] = dt.dt.hour

            for feature_name, values in features.items():

                self.df[feature_name] = values

                self.feature_steps.append(

                    FeatureEngineeringStep(

                        feature_name=feature_name,

                        source_column=column,

                        transformation="Datetime",

                        description=f"Derived from {column}"

                    )

                )

            self.audit_log.append(

                f"Engineered datetime features from '{column}'."

            )

    # --------------------------------------------------
    # Automatic Encoding Strategy
    # --------------------------------------------------

    def _build_encoder(self):
        """
        Select the most suitable encoder for
        categorical features.
        """

        categorical = [

            c

            for c in self.schema.profile.categorical_columns

            if c != self.target

        ]

        if not categorical:
            return None, categorical

        cardinality = max(

            self.df[c].nunique(dropna=True)

            for c in categorical

        )

        if cardinality <= HIGH_CARDINALITY:

            encoder = OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False

            )

            strategy = "OneHotEncoder"

        else:

            encoder = OrdinalEncoder(

                handle_unknown="use_encoded_value",

                unknown_value=-1

            )

            strategy = "OrdinalEncoder"

        self.audit_log.append(

            f"Encoding strategy selected: {strategy}"

        )

        return encoder, categorical

    # --------------------------------------------------
    # Automatic Scaling Strategy
    # --------------------------------------------------

    def _build_scaler(self):
        """
        Select an appropriate scaler for
        numeric features.
        """

        numeric = [

            c

            for c in self.schema.profile.numeric_columns

            if c != self.target

        ]

        if not numeric:

            return None, numeric

        skew = []

        outlier_votes = 0

        for column in numeric:

            series = self.df[column].dropna()

            if len(series) < 10:
                continue

            skew.append(abs(series.skew()))

            q1 = series.quantile(0.25)

            q3 = series.quantile(0.75)

            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            ratio = (

                ((series < lower) | (series > upper))

                .mean()

            )

            if ratio > 0.05:

                outlier_votes += 1

        if outlier_votes > max(1, len(numeric) // 3):

            scaler = RobustScaler()

            strategy = "RobustScaler"

        elif np.mean(skew) > 2:

            scaler = PowerTransformer()

            strategy = "PowerTransformer"

        else:

            scaler = StandardScaler()

            strategy = "StandardScaler"

        self.audit_log.append(

            f"Scaling strategy selected: {strategy}"

        )

        return scaler, numeric

    # --------------------------------------------------
    # Automatic Imputation
    # --------------------------------------------------

    def _build_imputer(
        self,
        columns: List[str]
    ):
        """
        Create an imputer appropriate for the
        supplied columns.
        """

        if not columns:
            return None

        missing = self.df[columns].isna().mean().mean()

        if missing < 0.05:

            return SimpleImputer(
                strategy="median"
            )

        if missing < 0.20:

            return KNNImputer()

        return SimpleImputer(
            strategy="most_frequent"
        )

    # --------------------------------------------------
    # Column Transformer
    # --------------------------------------------------

    def _build_transformer(self):
        """
        Assemble the preprocessing transformer.
        """

        encoder, categorical = self._build_encoder()

        scaler, numeric = self._build_scaler()

        transformers = []

        if numeric:

            numeric_pipeline = Pipeline(

                [

                    (

                        "imputer",

                        self._build_imputer(numeric)

                    ),

                    (

                        "scaler",

                        scaler

                    )

                ]

            )

            transformers.append(

                (

                    "numeric",

                    numeric_pipeline,

                    numeric

                )

            )

        if categorical:

            categorical_pipeline = Pipeline(

                [

                    (

                        "imputer",

                        SimpleImputer(

                            strategy="most_frequent"

                        )

                    ),

                    (

                        "encoder",

                        encoder

                    )

                ]

            )

            transformers.append(

                (

                    "categorical",

                    categorical_pipeline,

                    categorical

                )

            )

        transformer = ColumnTransformer(

            transformers=transformers,

            remainder="drop"

        )

        self.audit_log.append(

            "ColumnTransformer created."

        )

        return transformer

    # --------------------------------------------------
    # Apply Feature Engineering
    # --------------------------------------------------
    def _feature_engineering(self) -> None:
        """
        Execute all feature engineering stages.
        """

        self._engineer_datetime_features()

        self._time_series_features()

        self._cyclical_encoding()

        self._prevent_leakage()

        self._remove_correlated_features()

        self._remove_low_variance()

        self.feature_ranking = self._rank_features()

        self.audit_log.append(
            "AI feature engineering completed."
        )
    # --------------------------------------------------
    # Remove Highly Correlated Features
    # --------------------------------------------------

    def _remove_correlated_features(self) -> None:
        """
        Remove highly correlated numeric features while
        preserving the target column.
        """

        logger.info("Removing highly correlated features...")

        numeric = [
            c
            for c in self.schema.profile.numeric_columns
            if c != self.target and c in self.df.columns
        ]

        if len(numeric) < 2:
            return

        corr = self.df[numeric].corr().abs()

        upper = corr.where(
            np.triu(
                np.ones(corr.shape),
                k=1
            ).astype(bool)
        )

        to_drop = [
            column
            for column in upper.columns
            if any(
                upper[column] > CORRELATION_THRESHOLD
            )
        ]

        if not to_drop:
            return

        self.df.drop(
            columns=to_drop,
            inplace=True,
            errors="ignore"
        )

        for column in to_drop:

            self.audit_log.append(
                f"Removed correlated feature '{column}'."
            )

    # --------------------------------------------------
    # Low Variance Filter
    # --------------------------------------------------

    def _remove_low_variance(self) -> None:
        """
        Remove features with almost no variance.
        """

        logger.info("Removing low variance features...")

        numeric = [

            c

            for c in self.df.select_dtypes(
                include=np.number
            ).columns

            if c != self.target

        ]

        if len(numeric) < 2:
            return

        selector = VarianceThreshold(
            threshold=LOW_VARIANCE_THRESHOLD
        )

        selector.fit(self.df[numeric])

        keep = selector.get_support()

        removed = [

            col

            for col, retain in zip(numeric, keep)

            if not retain

        ]

        if removed:

            self.df.drop(
                columns=removed,
                inplace=True,
                errors="ignore"
            )

            for column in removed:

                self.audit_log.append(
                    f"Removed low variance feature '{column}'."
                )

    # --------------------------------------------------
    # Time Series Features
    # --------------------------------------------------

    def _time_series_features(self) -> None:
        """
        Generate lag and rolling statistics for
        time-series datasets.
        """

        if not self.schema.detect_time_series():
            return

        logger.info(
            "Generating time-series features..."
        )

        timestamp = (
            self.schema.detect_timestamp_column()
        )

        if timestamp is None:
            return

        numeric = [

            c

            for c in self.schema.profile.numeric_columns

            if c != self.target

        ]

        self.df.sort_values(
            timestamp,
            inplace=True
        )

        for column in numeric:

            lag = f"{column}_lag1"

            self.df[lag] = self.df[column].shift(1)

            self.feature_steps.append(

                FeatureEngineeringStep(

                    feature_name=lag,

                    source_column=column,

                    transformation="Lag",

                    description="Lag-1 feature"

                )

            )

            rolling = f"{column}_rolling_mean"

            self.df[rolling] = (

                self.df[column]

                .rolling(5)

                .mean()

            )

            self.feature_steps.append(

                FeatureEngineeringStep(

                    feature_name=rolling,

                    source_column=column,

                    transformation="Rolling Mean",

                    description="Window=5"

                )

            )

        self.audit_log.append(
            "Generated time-series features."
        )

    # --------------------------------------------------
    # Cyclical Encoding
    # --------------------------------------------------

    def _cyclical_encoding(self) -> None:
        """
        Encode cyclical calendar features.
        """

        logger.info("Applying cyclical encoding...")

        mappings = {

            "_month": 12,

            "_weekday": 7,

            "_hour": 24,

            "_day": 31

        }

        for column in list(self.df.columns):

            for suffix, period in mappings.items():

                if not column.endswith(suffix):
                    continue

                radians = (
                    2
                    * np.pi
                    * self.df[column]
                    / period
                )

                self.df[f"{column}_sin"] = np.sin(
                    radians
                )

                self.df[f"{column}_cos"] = np.cos(
                    radians
                )

                self.audit_log.append(
                    f"Cyclical encoding applied to '{column}'."
                )

    # --------------------------------------------------
    # Mutual Information Ranking
    # --------------------------------------------------

    def _rank_features(self) -> pd.DataFrame:
        """
        Rank predictive features using mutual information.
        """

        logger.info("Ranking predictive features...")

        if self.target not in self.df.columns:

            return pd.DataFrame()

        X = self.df.drop(
            columns=[self.target]
        )

        X = X.select_dtypes(
            include=np.number
        )

        if X.empty:

            return pd.DataFrame()

        y = self.df[self.target]

        problem = (
            self.problem_detector
            .analyze()
            .problem_type
        )

        if "Regression" in problem:

            scores = mutual_info_regression(
                X.fillna(0),
                y
            )

        else:

            scores = mutual_info_classif(
                X.fillna(0),
                y
            )

        ranking = pd.DataFrame(

            {

                "Feature": X.columns,

                "Importance": scores

            }

        )

        ranking.sort_values(

            "Importance",

            ascending=False,

            inplace=True

        )

        self.audit_log.append(
            "Feature importance ranking generated."
        )

        return ranking

    # --------------------------------------------------
    # Leakage Prevention
    # --------------------------------------------------

    def _prevent_leakage(self) -> None:
        """
        Remove identifier features and any columns
        explicitly flagged as leakage risks.
        """

        logger.info("Checking for leakage...")

        leakage = (
            self.problem_detector
            .detect_data_leakage()
        )

        removable = set(
            leakage +
            self.schema.profile.identifier_columns
        )

        removable.discard(self.target)

        removable = [

            c

            for c in removable

            if c in self.df.columns

        ]

        if removable:

            self.df.drop(
                columns=removable,
                inplace=True,
                errors="ignore"
            )

            for column in removable:

                self.audit_log.append(
                    f"Removed leakage feature '{column}'."
                )
    # --------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------

    def _split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Automatically determine the most appropriate
        train/test splitting strategy.
        """

        logger.info("Splitting dataset...")

        problem = self.problem_detector.analyze()

        # -----------------------------
        # Time Series
        # -----------------------------

        if self.schema.detect_time_series():

            splitter = TimeSeriesSplit(
                n_splits=5
            )

            train_idx, test_idx = list(splitter.split(X))[-1]

            self.audit_log.append(
                "TimeSeriesSplit selected."
            )

            return (

                X.iloc[train_idx],
                X.iloc[test_idx],

                y.iloc[train_idx],
                y.iloc[test_idx]

            )

        # -----------------------------
        # Classification
        # -----------------------------

        if "Classification" in problem.problem_type:

            self.audit_log.append(
                "Stratified train/test split selected."
            )

            return train_test_split(

                X,

                y,

                test_size=DEFAULT_TEST_SIZE,

                random_state=DEFAULT_RANDOM_STATE,

                stratify=y

            )

        # -----------------------------
        # Regression
        # -----------------------------

        self.audit_log.append(
            "Standard train/test split selected."
        )

        return train_test_split(

            X,

            y,

            test_size=DEFAULT_TEST_SIZE,

            random_state=DEFAULT_RANDOM_STATE

        )

    # --------------------------------------------------
    # Apply Transformer
    # --------------------------------------------------

    def _fit_transform_pipeline(
        self,
        transformer: ColumnTransformer,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ):
        """
        Fit and transform the preprocessing pipeline.
        """

        logger.info(
            "Fitting preprocessing pipeline..."
        )

        X_train_processed = transformer.fit_transform(
            X_train
        )

        X_test_processed = transformer.transform(
            X_test
        )

        self.audit_log.append(
            "ColumnTransformer fitted."
        )

        return (

            X_train_processed,

            X_test_processed

        )

    # --------------------------------------------------
    # Feature Names
    # --------------------------------------------------

    def _feature_names(
        self,
        transformer: ColumnTransformer
    ) -> List[str]:
        """
        Safely retrieve transformed feature names.
        """

        try:

            return list(

                transformer.get_feature_names_out()

            )

        except Exception:

            names = []

            for name, _, cols in transformer.transformers_:

                names.extend(cols)

            return names

    # --------------------------------------------------
    # Build Report
    # --------------------------------------------------

    def _report(self) -> PreprocessingReport:
        """
        Generate preprocessing report.
        """

        warnings = []

        if len(self.audit_log) > 20:

            warnings.append(
                "Large number of preprocessing steps applied."
            )

        if len(self.feature_steps) > 50:

            warnings.append(
                "Extensive feature engineering performed."
            )

        return PreprocessingReport(

            pipeline_version=self.pipeline_version,

            actions=self.audit_log.copy(),

            engineered_features=[

                step.feature_name

                for step in self.feature_steps

            ],

            warnings=warnings,

            statistics={

                "rows":

                    len(self.df),

                "columns":

                    len(self.df.columns),

                "engineered_features":

                    len(self.feature_steps)

            }

        )

    # --------------------------------------------------
    # Execute Model Preparation
    # --------------------------------------------------

    def _prepare_model_data(
        self,
        transformer: ColumnTransformer
    ):
        """
        Produce model-ready train and test datasets.
        """

        logger.info(
            "Preparing model-ready dataset..."
        )

        X = self.df.drop(
            columns=[self.target]
        )

        y = self.df[self.target]

        (

            X_train,

            X_test,

            y_train,

            y_test

        ) = self._split_dataset(
            X,
            y
        )

        (

            X_train_processed,

            X_test_processed

        ) = self._fit_transform_pipeline(

            transformer,

            X_train,

            X_test

        )

        names = self._feature_names(
            transformer
        )

        return (

            X_train,

            X_test,

            y_train,

            y_test,

            X_train_processed,

            X_test_processed,

            names

        )
    # --------------------------------------------------
    # Dataset Fingerprint
    # --------------------------------------------------

    def _dataset_fingerprint(self) -> str:
        """
        Generate a stable fingerprint for the dataset.
        """

        return hashlib.md5(

            pd.util.hash_pandas_object(

                self.df,

                index=True

            ).values

        ).hexdigest()

    # --------------------------------------------------
    # Build Bundle
    # --------------------------------------------------

    def build_bundle(
        self,
        result: PreprocessingResult
    ) -> PipelineBundle:
        """
        Build a reusable preprocessing bundle.
        """

        return PipelineBundle(

            pipeline_version=self.pipeline_version,

            dataset_fingerprint=self._dataset_fingerprint(),

            target_column=self.target,

            problem_type=self.problem_detector
            .analyze()
            .problem_type,

            feature_names=result.feature_names,

            preprocessing_pipeline=result.pipeline,

            metadata=result.metadata,

            report=result.report,

            audit_log=result.audit_log

        )

    # --------------------------------------------------
    # Executive Summary
    # --------------------------------------------------

    def executive_summary(self) -> str:
        """
        Human-readable preprocessing summary.
        """

        lines = []

        lines.append(
            "AI Adaptive Preprocessing Summary"
        )

        lines.append("=" * 40)

        lines.append(
            f"Pipeline Version: {self.pipeline_version}"
        )

        lines.append(
            f"Rows: {len(self.df)}"
        )

        lines.append(
            f"Columns: {len(self.df.columns)}"
        )

        lines.append(
            f"Target: {self.target}"
        )

        lines.append(
            f"Problem: "
            f"{self.problem_detector.analyze().problem_type}"
        )

        lines.append("")

        lines.append(
            f"Feature Engineering Steps: "
            f"{len(self.feature_steps)}"
        )

        lines.append(
            f"Pipeline Actions: "
            f"{len(self.audit_log)}"
        )

        return "\n".join(lines)

    # --------------------------------------------------
    # Export
    # --------------------------------------------------

    def to_dict(
        self,
        result: PreprocessingResult
    ) -> Dict[str, Any]:

        return {

            "metadata":

                result.metadata,

            "feature_names":

                result.feature_names,

            "pipeline_version":

                self.pipeline_version,

            "audit_log":

                result.audit_log,

            "statistics":

                result.report.statistics

        }

    # --------------------------------------------------

    def to_json(
        self,
        result: PreprocessingResult
    ) -> str:

        return json.dumps(

            self.to_dict(result),

            indent=4,

            default=str

        )
# ==========================================================
# Pipeline Bundle
# ==========================================================

from dataclasses import asdict
from datetime import datetime
import json
import joblib


@dataclass
class PipelineBundle:
    """
    Complete preprocessing bundle that can be saved,
    loaded and reused for future inference.
    """

    pipeline_version: str

    dataset_fingerprint: str

    target_column: str

    problem_type: str

    feature_names: List[str]

    preprocessing_pipeline: Any

    metadata: Dict[str, Any]

    report: PreprocessingReport

    audit_log: List[str]

    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # --------------------------------------------------

    def save(
        self,
        directory: str
    ) -> None:
        """
        Persist the bundle to disk.
        """

        Path(directory).mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(

            self.preprocessing_pipeline,

            Path(directory) / "pipeline.joblib"

        )

        metadata = {

            "pipeline_version":
                self.pipeline_version,

            "dataset_fingerprint":
                self.dataset_fingerprint,

            "target_column":
                self.target_column,

            "problem_type":
                self.problem_type,

            "feature_names":
                self.feature_names,

            "metadata":
                self.metadata,

            "report":
                asdict(self.report),

            "audit_log":
                self.audit_log,

            "created_at":
                self.created_at

        }

        with open(

            Path(directory) / "bundle.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=4,

                default=str

            )

    # --------------------------------------------------

    @classmethod

    def load(
        cls,
        directory: str
    ):

        pipeline = joblib.load(

            Path(directory) / "pipeline.joblib"

        )

        with open(

            Path(directory) / "bundle.json",

            "r",

            encoding="utf-8"

        ) as file:

            metadata = json.load(file)

        report = PreprocessingReport(

            pipeline_version=metadata["report"]["pipeline_version"],

            actions=metadata["report"]["actions"],

            engineered_features=metadata["report"]["engineered_features"],

            warnings=metadata["report"]["warnings"],

            statistics=metadata["report"]["statistics"]

        )

        return cls(

            pipeline_version=metadata["pipeline_version"],

            dataset_fingerprint=metadata["dataset_fingerprint"],

            target_column=metadata["target_column"],

            problem_type=metadata["problem_type"],

            feature_names=metadata["feature_names"],

            preprocessing_pipeline=pipeline,

            metadata=metadata["metadata"],

            report=report,

            audit_log=metadata["audit_log"],

            created_at=metadata["created_at"]

        )
