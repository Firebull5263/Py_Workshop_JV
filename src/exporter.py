"""
File export utilities.
"""

from __future__ import annotations

import io

import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
)

from reportlab.lib.styles import getSampleStyleSheet


class Exporter:

    @staticmethod
    def csv(df):

        return df.to_csv(index=False).encode()

    @staticmethod
    def excel(df):

        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl",
        ) as writer:

            df.to_excel(
                writer,
                index=False,
            )

        return output.getvalue()

    @staticmethod
    def pdf(result):

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(buffer)

        styles = getSampleStyleSheet()

        story = [

            Paragraph(
                "WeatherAI Model Report",
                styles["Heading1"],
            ),

            Paragraph(
                f"Problem Type: {result.problem_type}",
                styles["BodyText"],
            ),

            Paragraph(
                f"Target: {result.target}",
                styles["BodyText"],
            ),

        ]

        for k, v in result.metrics.items():

            story.append(

                Paragraph(
                    f"{k}: {v:.4f}",
                    styles["BodyText"],
                )

            )

        doc.build(story)

        buffer.seek(0)

        return buffer
