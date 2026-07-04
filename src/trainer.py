"""
Adaptive XGBoost training engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import (
    TimeSeriesSplit,
    KFold,
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)


@dataclass
class TrainingResult:

    model: object

    X_train: pd.DataFrame

    X_test: pd.DataFrame

    y_train: pd.Series

    y_test: pd.Series

    predictions: np.ndarray

    problem_type: str

    target: str

    feature_names: list

    best_params: dict

    metrics: dict


class ModelTrainer:

    def detect_problem(
        self,
        y: pd.Series,
    ) -> str:

        if y.dtype == object:

            return "classification"

        unique = y.nunique()

        if unique <= 10:

            return "classification"

        return "regression"

    def train(
        self,
        processed: dict,
    ) -> TrainingResult:

        df = processed["data"]

        target = processed["target"]

        timestamp = processed["timestamp"]

        X = df.drop(columns=[target])

        y = df[target]

        problem = self.detect_problem(y)

        if timestamp is not None:

            split = int(len(df) * 0.8)

            X_train = X.iloc[:split]

            X_test = X.iloc[split:]

            y_train = y.iloc[:split]

            y_test = y.iloc[split:]

            cv = TimeSeriesSplit(
                n_splits=5
            )

        else:

            stratify = (
                y
                if problem == "classification"
                else None
            )

            X_train, X_test, y_train, y_test = train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42,

                stratify=stratify,

            )

            if problem == "classification":

                cv = StratifiedKFold(

                    n_splits=5,

                    shuffle=True,

                    random_state=42,

                )

            else:

                cv = KFold(

                    n_splits=5,

                    shuffle=True,

                    random_state=42,

                )

        if problem == "classification":

            model = XGBClassifier(

                random_state=42,

                eval_metric="logloss",

                n_jobs=-1,

            )

            scoring = "accuracy"

        else:

            model = XGBRegressor(

                objective="reg:squarederror",

                random_state=42,

                n_jobs=-1,

            )

            scoring = "neg_root_mean_squared_error"

        params = {

            "n_estimators": [100, 200, 300],

            "max_depth": [3, 5, 7],

            "learning_rate": [0.01, 0.05, 0.1],

            "subsample": [0.8, 1.0],

            "colsample_bytree": [0.8, 1.0],

        }

        progress = st.progress(0)

        status = st.empty()

        status.info("Training XGBoost...")

        search = RandomizedSearchCV(

            estimator=model,

            param_distributions=params,

            n_iter=10,

            scoring=scoring,

            cv=cv,

            random_state=42,

            n_jobs=-1,

        )

        search.fit(

            X_train,

            y_train,

        )

        progress.progress(100)

        status.success("Training completed.")

        best = search.best_estimator_

        predictions = best.predict(X_test)

        if problem == "classification":

            metrics = {

                "Accuracy": accuracy_score(

                    y_test,

                    predictions,

                ),

                "Precision": precision_score(

                    y_test,

                    predictions,

                    average="weighted",

                    zero_division=0,

                ),

                "Recall": recall_score(

                    y_test,

                    predictions,

                    average="weighted",

                    zero_division=0,

                ),

                "F1": f1_score(

                    y_test,

                    predictions,

                    average="weighted",

                    zero_division=0,

                ),

            }

        else:

            rmse = np.sqrt(

                mean_squared_error(

                    y_test,

                    predictions,

                )

            )

            metrics = {

                "MAE": mean_absolute_error(

                    y_test,

                    predictions,

                ),

                "RMSE": rmse,

                "R²": r2_score(

                    y_test,

                    predictions,

                ),

            }

        st.success(

            f"Detected Problem Type: {problem.title()}"

        )

        return TrainingResult(

            model=best,

            X_train=X_train,

            X_test=X_test,

            y_train=y_train,

            y_test=y_test,

            predictions=predictions,

            problem_type=problem,

            target=target,

            feature_names=list(X.columns),

            best_params=search.best_params_,

            metrics=metrics,

        )
