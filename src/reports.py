"""
Downloadable reports.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.exporter import Exporter


class ReportGenerator:

    def render(self, result):

        st.header("Reports")

        predictions = pd.DataFrame(
            {
                "Actual": result.y_test,
                "Prediction": result.predictions,
            }
        )

        metrics = pd.DataFrame(
            result.metrics.items(),
            columns=[
                "Metric",
                "Value",
            ],
        )

        st.subheader(
            "Available Downloads"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.download_button(

                "Prediction CSV",

                Exporter.csv(
                    predictions
                ),

                "predictions.csv",

                "text/csv",
            )

            st.download_button(

                "Metrics CSV",

                Exporter.csv(
                    metrics
                ),

                "metrics.csv",

                "text/csv",
            )

        with c2:

            st.download_button(

                "Prediction Excel",

                Exporter.excel(
                    predictions
                ),

                "predictions.xlsx",

            )

            st.download_button(

                "PDF Report",

                Exporter.pdf(
                    result
                ),

                "report.pdf",

            )

        st.success(
            "Reports ready for download."
        )
