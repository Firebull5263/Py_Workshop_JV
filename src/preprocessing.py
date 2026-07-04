"""
Adaptive preprocessing engine.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.logger import get_logger
from src.utils import (
    guess_target,
    guess_timestamp,
    is_identifier,
)


class AdaptivePreprocessor:

    def __init__(self):

        self.logger = get_logger(__name__)

    def _parse_dates(
        self,
        df: pd.DataFrame,
        timestamp: str,
    ) -> pd.DataFrame:

        df = df.copy()

        df[timestamp] = pd.to_datetime(
            df[timestamp],
            errors="coerce",
        )

        df["Year"] = df[timestamp].dt.year
        df["Month"] = df[timestamp].dt.month
        df["Day"] = df[timestamp].dt.day
        df["Hour"] = df[timestamp].dt.hour
        df["DayOfWeek"] = (
            df[timestamp].dt.dayofweek
        )

        return df

    def run(
        self,
        df: pd.DataFrame,
    ) -> dict:

        df = df.copy()

        timestamp = guess_timestamp(df)

        target = guess_target(df)

        if timestamp:

            df = self._parse_dates(
                df,
                timestamp,
            )

        removed_duplicates = (
            len(df) - len(df.drop_duplicates())
        )

        df = df.drop_duplicates()

        identifiers = [

            c

            for c in df.columns

            if is_identifier(df[c])

            and c != target
        ]

        if identifiers:

            df = df.drop(
                columns=identifiers
            )

        numeric = list(
            df.select_dtypes(
                include="number"
            ).columns
        )

        categorical = [

            c

            for c in df.select_dtypes(
                include="object"
            ).columns

            if c != target
        ]

        num_imputer = SimpleImputer(
            strategy="median"
        )

        cat_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        df[numeric] = num_imputer.fit_transform(
            df[numeric]
        )

        if categorical:

            df[categorical] = cat_imputer.fit_transform(
                df[categorical]
            )

            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )

            encoded = encoder.fit_transform(
                df[categorical]
            )

            encoded = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(
                    categorical
                ),
                index=df.index,
            )

            df = df.drop(
                columns=categorical
            )

            df = pd.concat(
                [df, encoded],
                axis=1,
            )

        features = [
            c
            for c in df.columns
            if c != target
        ]

        scaler = StandardScaler()

        df[features] = scaler.fit_transform(
            df[features]
        )

        st.success(
            "Automatic preprocessing completed."
        )

        with st.expander(
            "Preprocessing Summary",
            expanded=True,
        ):

            st.markdown(
                f"""
- Target detected: **{target}**
- Timestamp detected: **{timestamp}**
- Duplicate rows removed: **{removed_duplicates}**
- Identifier columns removed: **{len(identifiers)}**
- Missing numeric values: Median imputation
- Missing categorical values: Most Frequent imputation
- Categorical encoding: One-Hot Encoding
- Feature scaling: StandardScaler
"""
            )

        return {

            "data": df,

            "target": target,

            "timestamp": timestamp,

            "features": features,
        }
