from pathlib import Path

import streamlit as st
import traceback

from src.detector import DatasetDetector
from src.preprocessing import AdaptivePreprocessor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import Visualizer
from src.reports import ReportGenerator

st.set_page_config(
    page_title="WeatherAI",
    page_icon="🌦️",
    layout="wide",
)

css = Path("assets/styles.css")

if css.exists():
    st.markdown(
        f"<style>{css.read_text()}</style>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='main-title'>🤖 WeatherAI</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Adaptive XGBoost Machine Learning Platform"
)

uploaded = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"],
)

if uploaded is None:

    st.info(
        "Upload a CSV dataset to begin."
    )

    st.stop()

detector = DatasetDetector()

dataset = detector.load_dataset(uploaded)

st.success("Dataset Loaded")

tabs = st.tabs(
    [
        "📊 Dataset",
        "🧹 Preprocessing",
        "🤖 Training",
        "📈 Evaluation",
        "📉 Visualisations",
        "📄 Reports",
    ]
)

with tabs[0]:

    detector.render(dataset)

with tabs[1]:

    preprocessor = AdaptivePreprocessor()

    processed = preprocessor.run(dataset)

with tabs[2]:

    try:

        trainer = ModelTrainer()

        training_result = trainer.train(processed)

    except Exception as ex:

        st.error(
            "Model training failed."
        )

    with st.expander(
        "Technical Details"
    ):
        st.code(traceback.format_exc())

    st.stop()

with tabs[3]:

    evaluator = ModelEvaluator()

    evaluator.render(training_result)

with tabs[4]:

    visualizer = Visualizer()

    visualizer.render(training_result)

with tabs[5]:

    reports = ReportGenerator()

    reports.render(training_result)

st.markdown(
"""
<div class='footer'>
Built with Streamlit • XGBoost • Plotly
</div>
""",
unsafe_allow_html=True,
)
