"""
Interactive Plotly visualisations.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


class Visualizer:

    def render(self, result) -> None:

        st.header("Interactive Visualisations")

        df = result.X_test.copy()

        df["Prediction"] = result.predictions

        df["Actual"] = result.y_test.values

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Importance",
                "Distributions",
                "Correlation",
                "Predictions",
            ]
        )

        # ------------------------------------------------ #

        with tab1:

            if hasattr(result.model, "feature_importances_"):

                importance = pd.DataFrame(
                    {
                        "Feature": result.feature_names,
                        "Importance": result.model.feature_importances_,
                    }
                ).sort_values(
                    "Importance",
                    ascending=False,
                )

                fig = px.bar(
                    importance.head(20),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Feature Importance",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        # ------------------------------------------------ #

        with tab2:

            numeric = df.select_dtypes(
                include="number"
            ).columns

            column = st.selectbox(
                "Distribution",
                numeric,
            )

            fig = px.histogram(
                df,
                x=column,
                nbins=30,
                marginal="box",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ------------------------------------------------ #

        with tab3:

            corr = df.select_dtypes(
                include="number"
            ).corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Matrix",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ------------------------------------------------ #

        with tab4:

            if result.problem_type == "regression":

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        y=result.y_test,
                        mode="lines",
                        name="Actual",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        y=result.predictions,
                        mode="lines",
                        name="Prediction",
                    )
                )

                fig.update_layout(
                    title="Actual vs Prediction"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                compare = pd.DataFrame(
                    {
                        "Actual": result.y_test,
                        "Prediction": result.predictions,
                    }
                )

                st.dataframe(compare.head(50))
