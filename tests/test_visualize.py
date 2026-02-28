"""Testes do módulo src/visualize.py."""
import os

import numpy as np
import pandas as pd
import pytest

from src.visualize import (
    plot_comparison,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_consolidated,
)


class TestPlotComparison:
    def test_creates_file(self, results_df, tmp_path):
        filepath = str(tmp_path / "comparison.png")
        plot_comparison(results_df, "Test", filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0


class TestPlotRocCurves:
    def test_creates_file(self, synthetic_df, feature_columns, trained_models_dict, tmp_path):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        filepath = str(tmp_path / "roc.png")
        plot_roc_curves(trained_models_dict, X, y, "Test", filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0


class TestPlotConfusionMatrices:
    def test_creates_file(self, synthetic_df, feature_columns, trained_models_dict, tmp_path):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        filepath = str(tmp_path / "confusion.png")
        plot_confusion_matrices(trained_models_dict, X, y, "Test", filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

    def test_single_model(self, synthetic_df, feature_columns, trained_models_dict, tmp_path):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        single = {k: v for k, v in list(trained_models_dict.items())[:1]}
        filepath = str(tmp_path / "confusion_single.png")
        plot_confusion_matrices(single, X, y, "Test", filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0


class TestPlotConsolidated:
    def test_creates_file(self, tmp_path):
        summary = pd.DataFrame({
            "Model": ["GB", "RF", "GB", "RF", "GB", "RF"],
            "Accuracy": [0.85, 0.80, 0.83, 0.78, 0.82, 0.77],
            "Sensitivity": [0.82, 0.78, 0.80, 0.76, 0.79, 0.75],
            "Specificity": [0.88, 0.83, 0.86, 0.81, 0.85, 0.80],
            "AUC": [0.90, 0.86, 0.88, 0.84, 0.87, 0.83],
            "Etapa": ["Treino", "Treino", "Teste", "Teste", "Validação", "Validação"],
        })
        filepath = str(tmp_path / "consolidated.png")
        plot_consolidated(summary, filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
