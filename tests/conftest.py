"""Fixtures compartilhadas para os testes."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


@pytest.fixture
def synthetic_df():
    """DataFrame sintético com 100 linhas mimicando a estrutura dos dados reais."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "ra": range(1000, 1000 + n),
        "idade": np.random.uniform(8, 18, n),
        "genero.masculino": np.random.randint(0, 2, n),
        "instituicao.publica": np.random.randint(0, 2, n),
        "nº.av": np.random.randint(1, 5, n).astype(float),
        "iaa": np.random.uniform(0, 10, n),
        "ieg": np.random.uniform(0, 10, n),
        "ips": np.random.uniform(0, 10, n),
        "ida": np.random.uniform(0, 10, n),
        "ipv": np.random.uniform(0, 10, n),
        "ian": np.random.uniform(0, 10, n),
        "matematica": np.random.uniform(0, 10, n),
        "fase_0": np.random.randint(0, 2, n),
        "fase_1": np.random.randint(0, 2, n),
        "fase_2": np.random.randint(0, 2, n),
        "fase_3": np.random.randint(0, 2, n),
        "defasagem": np.random.randint(0, 2, n),
    })


@pytest.fixture
def feature_columns():
    """Lista de colunas de features (sem ra e defasagem)."""
    return [
        "idade", "genero.masculino", "instituicao.publica",
        "nº.av", "iaa", "ieg", "ips", "ida", "ipv", "ian", "matematica",
        "fase_0", "fase_1", "fase_2", "fase_3",
    ]


@pytest.fixture
def trained_model(synthetic_df, feature_columns):
    """GradientBoostingClassifier treinado em dados sintéticos."""
    X = synthetic_df[feature_columns]
    y = synthetic_df["defasagem"]
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_models_dict(synthetic_df, feature_columns):
    """Dict com 2 modelos treinados para testes de evaluate e plot."""
    X = synthetic_df[feature_columns]
    y = synthetic_df["defasagem"]
    gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gb.fit(X, y)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    return {"Gradient Boosting": gb, "Random Forest": rf}


@pytest.fixture
def results_df():
    """DataFrame de métricas no formato esperado pelas funções de plot."""
    return pd.DataFrame({
        "Model": ["Gradient Boosting", "Random Forest"],
        "Accuracy": [0.85, 0.80],
        "Sensitivity": [0.82, 0.78],
        "Specificity": [0.88, 0.83],
        "AUC": [0.90, 0.86],
        "F1": [0.83, 0.79],
        "Precision": [0.84, 0.80],
    })


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Diretório temporário para salvar arquivos de saída."""
    output = tmp_path / "output"
    output.mkdir()
    return output
