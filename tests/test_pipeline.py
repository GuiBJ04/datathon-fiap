"""Testes unitários do projeto."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import _normalize_colname, _encode_instituicao
from src.train import compute_metrics, split_data


class TestNormalizeColname:
    def test_lowercase(self):
        assert _normalize_colname("Idade 22") == "idade.22"

    def test_accent(self):
        assert _normalize_colname("Gênero") == "genero"

    def test_spaces(self):
        assert _normalize_colname("Instituição de ensino") == "instituicao.de.ensino"


class TestEncodeInstituicao:
    def test_publica(self):
        assert _encode_instituicao("Escola Pública") == 1

    def test_particular(self):
        assert _encode_instituicao("Rede Decisão") == 0

    def test_nan(self):
        assert np.isnan(_encode_instituicao(np.nan))


class TestComputeMetrics:
    def test_returns_all_keys(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(50, 5), columns=list("ABCDE"))
        y = pd.Series(np.random.randint(0, 2, 50))
        model = GradientBoostingClassifier(n_estimators=10, random_state=42).fit(X, y)
        m = compute_metrics(model, X, y, "test")
        for k in ["Model", "Accuracy", "Sensitivity", "Specificity", "AUC"]:
            assert k in m

    def test_values_in_range(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(50, 5), columns=list("ABCDE"))
        y = pd.Series(np.random.randint(0, 2, 50))
        model = GradientBoostingClassifier(n_estimators=10, random_state=42).fit(X, y)
        m = compute_metrics(model, X, y, "test")
        for k in ["Accuracy", "Sensitivity", "Specificity", "AUC"]:
            assert 0 <= m[k] <= 1


class TestSplitData:
    def test_shapes(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.rand(100, 5), columns=list("ABCDE"))
        df["defasagem"] = np.random.randint(0, 2, 100)
        X_tr, X_te, y_tr, y_te = split_data(df, test_size=0.3)
        assert len(X_tr) == 70
        assert len(X_te) == 30

    def test_stratified(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.rand(100, 3), columns=list("ABC"))
        df["defasagem"] = [0]*80 + [1]*20
        X_tr, X_te, y_tr, y_te = split_data(df, test_size=0.3)
        # Proporções devem ser aproximadamente iguais
        assert abs(y_tr.mean() - y_te.mean()) < 0.05
