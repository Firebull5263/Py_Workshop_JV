"""
Model evaluation dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)


class ModelEvaluator:

    def render(
        self,
        result,
    ) -> None:

        st.header("Model Evaluation")

        cols = st.columns(
            len(result.metrics)
        )

        for col, (metric, value) in zip(

            cols,

            result.metrics.items(),

        ):

            col.metric(

                metric,

                f"{value:.4f}",

            )

        st.subheader(
            "Best Hyperparameters"
        )

        st.json(result.best_params)

        if result.problem_type == "regression":

            residuals = (

                result.y_test

                - result.predictions

            )

            df = pd.DataFrame(

                {

                    "Actual": result.y_test,

                    "Predicted": result.predictions,

                    "Residual": residuals,

                }

            )

            fig = px.scatter(

                df,

                x="Actual",

                y="Predicted",

                trendline="ols",

                title="Actual vs Predicted",

            )

            st.plotly_chart(

                fig,

                use_container_width=True,

            )

            fig2 = px.histogram(

                df,

                x="Residual",

                nbins=30,

                title="Residual Distribution",

            )

            st.plotly_chart(

                fig2,

                use_container_width=True,

            )

        else:

            cm = confusion_matrix(

                result.y_test,

                result.predictions,

            )

            fig = px.imshow(

                cm,

                text_auto=True,

                title="Confusion Matrix",

            )

            st.plotly_chart(

                fig,

                use_container_width=True,

            )

            try:

                probabilities = result.model.predict_proba(

                    result.X_test

                )[:, 1]

                fpr, tpr, _ = roc_curve(

                    result.y_test,

                    probabilities,

                )

                roc_auc = auc(

                    fpr,

                    tpr,

                )

                fig = go.Figure()

                fig.add_trace(

                    go.Scatter(

                        x=fpr,

                        y=tpr,

                        mode="lines",

                        name=f"AUC={roc_auc:.3f}",

                    )

                )

                fig.update_layout(

                    title="ROC Curve",

                    xaxis_title="False Positive Rate",

                    yaxis_title="True Positive Rate",

                )

                st.plotly_chart(

                    fig,

                    use_container_width=True,

                )

            except Exception:

                st.info(

                    "ROC curve unavailable for this dataset."

                )

        with st.expander(

            "Prediction Sample",

            expanded=False,

        ):

            preview = pd.DataFrame(

                {

                    "Actual": result.y_test,

                    "Prediction": result.predictions,

                }

            )

            st.dataframe(

                preview.head(25),

                use_container_width=True,

            )
