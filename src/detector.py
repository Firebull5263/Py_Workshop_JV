"""
Dataset inspection and automatic metadata detection.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.logger import get_logger
from src.utils import (
    guess_target,
    guess_timestamp,
    is_identifier,
)


class DatasetDetector:

    def __init__(self):

        self.logger = get_logger(__name__)

    @st.cache_data(show_spinner=False)
    def load_dataset(
        self,
        uploaded_file,
    ) -> pd.DataFrame:

        try:

            df = pd.read_csv(
                uploaded_file,
                encoding="utf-8",
            )
        except UnicodeDecodeError:

            df = pd.read_csv(
                uploaded_file,
                encoding="latin1",
            )

        if df.empty:

            raise ValueError(
                "Dataset contains no rows."
            )

        if len(df.columns) < 2:

            raise ValueError(
                "Dataset requires at least two columns."
            )

        df = df.drop_duplicates()

        return df
    def analyse(
        self,
        df: pd.DataFrame,
    ) -> dict:

        numeric = list(
            df.select_dtypes(
                include="number"
            ).columns
        )

        categorical = list(
            df.select_dtypes(
                include="object"
            ).columns
        )

        datetime_cols = list(
            df.select_dtypes(
                include="datetime"
            ).columns
        )

        identifiers = [
            c
            for c in df.columns
            if is_identifier(df[c])
        ]

        timestamp = guess_timestamp(df)

        target = guess_target(df)

        return {

            "numeric": numeric,

            "categorical": categorical,

            "datetime": datetime_cols,

            "identifier": identifiers,

            "timestamp": timestamp,

            "target": target,
        }

    def render(
        self,
        df: pd.DataFrame,
    ) -> None:

        info = self.analyse(df)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            len(df),
        )

        c2.metric(
            "Columns",
            len(df.columns),
        )

        c3.metric(
            "Missing Values",
            int(df.isna().sum().sum()),
        )

        st.subheader("Detected Dataset Information")

        st.json(info)

        st.subheader("Preview")

        st.dataframe(df.head())

        with st.expander(
            "Column Data Types"
        ):

            st.write(df.dtypes)
