"""Testes do módulo src/train.py."""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.train import (
    split_data,
    prepare_validation,
    train_all_models,
    save_models,
    evaluate_all,
    compute_metrics,
)


class TestPrepareValidation:
    def test_aligns_columns(self, synthetic_df, feature_columns):
        X_train = synthetic_df[feature_columns]
        val_df = synthetic_df.copy()
        val_df = val_df.drop(columns=["ra"])
        X_val, y_val = prepare_validation(val_df, X_train)
        assert list(X_val.columns) == list(X_train.columns)

    def test_adds_missing_cols(self, synthetic_df, feature_columns):
        X_train = synthetic_df[feature_columns].copy()
        X_train["extra_col"] = 1.0
        val_df = synthetic_df.drop(columns=["ra"]).copy()
        X_val, y_val = prepare_validation(val_df, X_train)
        assert "extra_col" in X_val.columns
        assert (X_val["extra_col"] == 0).all()

    def test_returns_correct_shapes(self, synthetic_df, feature_columns):
        X_train = synthetic_df[feature_columns]
        val_df = synthetic_df.drop(columns=["ra"]).copy()
        X_val, y_val = prepare_validation(val_df, X_train)
        assert len(X_val) == len(val_df)
        assert len(y_val) == len(val_df)


class TestTrainAllModels:
    @patch("src.train.GridSearchCV")
    def test_returns_4_models(self, mock_grid_cls, synthetic_df, feature_columns):
        mock_grid = MagicMock()
        mock_grid.best_score_ = 0.85
        mock_grid.best_params_ = {"n_estimators": 10}
        mock_grid.best_estimator_ = GradientBoostingClassifier(
            n_estimators=10, random_state=42
        )
        mock_grid_cls.return_value = mock_grid
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        results = train_all_models(X, y)
        assert len(results) == 4
        expected = {"Random Forest", "Logistic Regression", "Gradient Boosting", "Neural Network"}
        assert set(results.keys()) == expected

    @patch("src.train.GridSearchCV")
    def test_best_score(self, mock_grid_cls, synthetic_df, feature_columns):
        mock_grid = MagicMock()
        mock_grid.best_score_ = 0.90
        mock_grid.best_params_ = {}
        mock_grid.best_estimator_ = GradientBoostingClassifier(n_estimators=5)
        mock_grid_cls.return_value = mock_grid
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        results = train_all_models(X, y)
        for name, grid in results.items():
            assert grid.best_score_ == 0.90


class TestSaveModels:
    def test_creates_files(self, tmp_path):
        gb = GradientBoostingClassifier(n_estimators=5, random_state=42)
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randint(0, 2, 20))
        gb.fit(X, y)
        rf.fit(X, y)
        grid_results = {}
        for name, est in [("Gradient Boosting", gb), ("Random Forest", rf)]:
            mock_grid = MagicMock()
            mock_grid.best_estimator_ = est
            mock_grid.best_params_ = {"n_estimators": 5}
            grid_results[name] = mock_grid
        models = save_models(grid_results, str(tmp_path))
        assert (tmp_path / "gradient_boosting.pkl").exists()
        assert (tmp_path / "random_forest.pkl").exists()
        assert (tmp_path / "best_params.json").exists()

    def test_returns_estimators(self, tmp_path):
        gb = GradientBoostingClassifier(n_estimators=5, random_state=42)
        X = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randint(0, 2, 20))
        gb.fit(X, y)
        mock_grid = MagicMock()
        mock_grid.best_estimator_ = gb
        mock_grid.best_params_ = {"n_estimators": 5}
        grid_results = {"Gradient Boosting": mock_grid}
        models = save_models(grid_results, str(tmp_path))
        assert "Gradient Boosting" in models
        assert isinstance(models["Gradient Boosting"], GradientBoostingClassifier)

    def test_json_valid(self, tmp_path):
        gb = GradientBoostingClassifier(n_estimators=5, random_state=42)
        X = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randint(0, 2, 20))
        gb.fit(X, y)
        mock_grid = MagicMock()
        mock_grid.best_estimator_ = gb
        mock_grid.best_params_ = {"n_estimators": 5}
        save_models({"GB": mock_grid}, str(tmp_path))
        with open(tmp_path / "best_params.json") as f:
            data = json.load(f)
        assert "GB" in data


class TestEvaluateAll:
    def test_returns_dataframe(self, synthetic_df, feature_columns, trained_models_dict):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        df = evaluate_all(trained_models_dict, X, y, "Test")
        assert isinstance(df, pd.DataFrame)

    def test_has_etapa_column(self, synthetic_df, feature_columns, trained_models_dict):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        df = evaluate_all(trained_models_dict, X, y, "Treino")
        assert "Etapa" in df.columns
        assert (df["Etapa"] == "Treino").all()

    def test_num_rows(self, synthetic_df, feature_columns, trained_models_dict):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        df = evaluate_all(trained_models_dict, X, y, "Test")
        assert len(df) == len(trained_models_dict)

    def test_metric_columns(self, synthetic_df, feature_columns, trained_models_dict):
        X = synthetic_df[feature_columns]
        y = synthetic_df["defasagem"]
        df = evaluate_all(trained_models_dict, X, y, "Test")
        for col in ["Model", "Accuracy", "Sensitivity", "Specificity", "AUC"]:
            assert col in df.columns
