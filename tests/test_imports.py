from src.detector import DatasetDetector
from src.preprocessing import AdaptivePreprocessor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import Visualizer
from src.reports import ReportGenerator


def test_imports():

    DatasetDetector()

    AdaptivePreprocessor()

    ModelTrainer()

    ModelEvaluator()

    Visualizer()

    ReportGenerator()

    assert True
