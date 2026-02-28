"""Testes do módulo src/utils.py."""
import json
import logging
import pickle

import pytest

from src.utils import setup_logger, save_model, load_model, save_json, normalize_colname


class TestSetupLogger:
    def test_returns_logger(self):
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self):
        logger = setup_logger("test_logger_handler")
        assert len(logger.handlers) >= 1

    def test_logger_level(self):
        logger = setup_logger("test_logger_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers(self):
        name = "test_no_dup"
        logger1 = setup_logger(name)
        n_handlers = len(logger1.handlers)
        logger2 = setup_logger(name)
        assert len(logger2.handlers) == n_handlers


class TestSaveModel:
    def test_save_creates_file(self, tmp_path, monkeypatch):
        import src.utils as utils_mod
        monkeypatch.setattr(utils_mod, "MODEL_DIR", tmp_path)
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=2, random_state=42)
        path = save_model(model, "test_model.pkl")
        assert (tmp_path / "test_model.pkl").exists()
        assert path == str(tmp_path / "test_model.pkl")


class TestLoadModel:
    def test_load_model(self, tmp_path, monkeypatch):
        import src.utils as utils_mod
        monkeypatch.setattr(utils_mod, "MODEL_DIR", tmp_path)
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=2, random_state=42)
        save_model(model, "test_load.pkl")
        loaded = load_model("test_load.pkl")
        assert type(loaded) == type(model)

    def test_load_model_not_found(self, tmp_path, monkeypatch):
        import src.utils as utils_mod
        monkeypatch.setattr(utils_mod, "MODEL_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent.pkl")


class TestSaveJson:
    def test_save_json_creates_file(self, tmp_path, monkeypatch):
        import src.utils as utils_mod
        monkeypatch.setattr(utils_mod, "MODEL_DIR", tmp_path)
        data = {"accuracy": 0.95, "model": "GB"}
        path = save_json(data, "metrics.json")
        assert (tmp_path / "metrics.json").exists()

    def test_save_json_valid_content(self, tmp_path, monkeypatch):
        import src.utils as utils_mod
        monkeypatch.setattr(utils_mod, "MODEL_DIR", tmp_path)
        data = {"accuracy": 0.95, "model": "GB"}
        save_json(data, "metrics.json")
        with open(tmp_path / "metrics.json") as f:
            loaded = json.load(f)
        assert loaded == data
