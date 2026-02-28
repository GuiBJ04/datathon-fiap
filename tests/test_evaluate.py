"""Testes do módulo src/evaluate.py."""
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.evaluate import compute_metrics, evaluate_all, plot_comparison, plot_roc_curves, plot_consolidated


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])
    y = pd.Series((X["a"] > 0.5).astype(int))
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestComputeMetrics:
    def test_returns_dict(self, trained_model):
        model, X, y = trained_model
        result = compute_metrics(model, X, y, "test")
        assert isinstance(result, dict)

    def test_has_all_keys(self, trained_model):
        model, X, y = trained_model
        result = compute_metrics(model, X, y, "test")
        for key in ["Model", "Accuracy", "Sensitivity", "Specificity", "AUC"]:
            assert key in result

    def test_metrics_in_range(self, trained_model):
        model, X, y = trained_model
        result = compute_metrics(model, X, y, "test")
        for key in ["Accuracy", "Sensitivity", "Specificity", "AUC"]:
            assert 0 <= result[key] <= 1


class TestEvaluateAll:
    def test_returns_df(self, synthetic_df, feature_columns, trained_models_dict):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        df = evaluate_all(trained_models_dict, X, y, "Test")
        assert isinstance(df, pd.DataFrame)

    def test_num_rows(self, synthetic_df, feature_columns, trained_models_dict):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        df = evaluate_all(trained_models_dict, X, y, "Test")
        assert len(df) == len(trained_models_dict)

    def test_has_metric_columns(self, synthetic_df, feature_columns, trained_models_dict):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        df = evaluate_all(trained_models_dict, X, y, "Test")
        for col in ["Model", "Accuracy", "Sensitivity", "Specificity", "AUC"]:
            assert col in df.columns


class TestPlotComparison:
    def test_creates_file(self, results_df, tmp_path, monkeypatch):
        import src.evaluate as eval_mod
        monkeypatch.setattr(eval_mod, "OUTPUT_DIR", tmp_path)
        plot_comparison(results_df, "Test", "comparison.png")
        assert (tmp_path / "comparison.png").exists()
        assert (tmp_path / "comparison.png").stat().st_size > 0


class TestPlotRocCurves:
    def test_creates_file(self, synthetic_df, feature_columns, trained_models_dict, tmp_path, monkeypatch):
        import src.evaluate as eval_mod
        monkeypatch.setattr(eval_mod, "OUTPUT_DIR", tmp_path)
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        plot_roc_curves(trained_models_dict, X, y, "Test", "roc.png")
        assert (tmp_path / "roc.png").exists()
        assert (tmp_path / "roc.png").stat().st_size > 0


class TestPlotConsolidated:
    def test_creates_file(self, results_df, tmp_path, monkeypatch):
        import src.evaluate as eval_mod
        monkeypatch.setattr(eval_mod, "OUTPUT_DIR", tmp_path)
        train_df = results_df.copy()
        test_df = results_df.copy()
        val_df = results_df.copy()
        summary = plot_consolidated(train_df, test_df, val_df, "consolidated.png")
        assert (tmp_path / "consolidated.png").exists()

    def test_returns_summary(self, results_df, tmp_path, monkeypatch):
        import src.evaluate as eval_mod
        monkeypatch.setattr(eval_mod, "OUTPUT_DIR", tmp_path)
        train_df = results_df.copy()
        test_df = results_df.copy()
        val_df = results_df.copy()
        summary = plot_consolidated(train_df, test_df, val_df, "consolidated.png")
        assert isinstance(summary, pd.DataFrame)
        assert "Etapa" in summary.columns
        assert set(summary["Etapa"].unique()) == {"Treino", "Teste", "Validação"}
