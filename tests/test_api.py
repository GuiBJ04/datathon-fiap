"""Testes da API."""
import os, sys
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from app.main import app, app_state


@pytest.fixture(autouse=True)
def setup():
    np.random.seed(42)
    cols = ["idade", "genero.masculino", "instituicao.publica", "nº.av",
            "iaa", "ieg", "ips", "ida", "ipv", "ian", "matematica",
            "fase_0", "fase_1", "fase_2", "fase_3", "fase_4", "fase_5", "fase_6", "fase_7"]
    X = pd.DataFrame(np.random.rand(80, len(cols)), columns=cols)
    y = pd.Series(np.random.randint(0, 2, 80))
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    app_state["model"] = model
    app_state["start_time"] = datetime.now()
    app_state["prediction_count"] = 0
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def payload():
    return {"idade": 12, "genero_masculino": 1, "instituicao_publica": 1,
            "num_av": 3, "iaa": 7.5, "ieg": 6.0, "ips": 5.8,
            "ida": 4.5, "ipv": 6.2, "ian": 5.0, "matematica": 5.0, "fase": 3}


class TestRoot:
    def test_200(self, client):
        assert client.get("/").status_code == 200


class TestHealth:
    def test_healthy(self, client):
        r = client.get("/health").json()
        assert r["status"] == "healthy"
        assert r["model_loaded"] is True


class TestPredict:
    def test_200(self, client, payload):
        assert client.post("/predict", json=payload).status_code == 200

    def test_returns_prediction(self, client, payload):
        r = client.post("/predict", json=payload).json()
        assert r["prediction"] in [0, 1]
        assert 0 <= r["probability"] <= 1
        assert r["risk_level"] in ["Baixo", "Moderado", "Alto"]

    def test_increments_count(self, client, payload):
        client.post("/predict", json=payload)
        client.post("/predict", json=payload)
        assert app_state["prediction_count"] == 2
