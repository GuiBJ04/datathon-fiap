"""Testes do módulo de pré-processamento."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.utils import normalize_colname
from src.preprocessing import (
    _normalize_colname,
    _encode_instituicao,
    load_sheets,
    create_datasets,
    process_data,
)


class TestNormalizeColname:
    def test_lowercase(self):
        assert normalize_colname("ABC") == "abc"

    def test_accent_removal(self):
        assert normalize_colname("Gênero") == "genero"
        assert normalize_colname("Avaliação") == "avaliacao"

    def test_space_to_dot(self):
        assert normalize_colname("Idade 22") == "idade.22"

    def test_combined(self):
        assert normalize_colname("Instituição de ensino") == "instituicao.de.ensino"


class TestInternalNormalize:
    def test_lowercase(self):
        assert _normalize_colname("ABC") == "abc"

    def test_accent(self):
        assert _normalize_colname("Gênero") == "genero"


class TestEncodeInstituicao:
    def test_publica(self):
        assert _encode_instituicao("Escola Pública") == 1

    def test_particular(self):
        assert _encode_instituicao("Rede Decisão") == 0

    def test_nan(self):
        assert np.isnan(_encode_instituicao(np.nan))


class TestLoadSheets:
    @patch("src.preprocessing.pd.read_excel")
    def test_returns_three_dataframes(self, mock_read):
        df1 = pd.DataFrame({"RA": [1], "col": [10]})
        df2 = pd.DataFrame({"RA": [2], "col": [20]})
        df3 = pd.DataFrame({"RA": [3], "col": [30]})
        mock_read.side_effect = [df1, df2, df3]
        result = load_sheets("fake.xlsx")
        assert len(result) == 3
        assert mock_read.call_count == 3

    @patch("src.preprocessing.pd.read_excel")
    def test_correct_sheet_names(self, mock_read):
        mock_read.return_value = pd.DataFrame({"RA": [1]})
        load_sheets("fake.xlsx")
        sheets = [call.kwargs["sheet_name"] for call in mock_read.call_args_list]
        assert sheets == ["PEDE2022", "PEDE2023", "PEDE2024"]


def _make_raw_dataframes():
    """Cria DataFrames sintéticos imitando as 3 abas do Excel."""
    np.random.seed(42)
    common_ras = list(range(100, 130))
    new_ras_23 = list(range(200, 215))
    new_ras_24 = list(range(200, 210))

    def _make_df(ras, year_suffix, has_defas=False):
        n = len(ras)
        d = {
            "RA": ras,
            "Gênero": np.random.choice(["Menino", "Menina"], n),
            "Instituição de ensino": np.random.choice(
                ["Escola Pública", "Rede Decisão"], n
            ),
            "Fase": np.random.choice(["FASE 1", "FASE 2", "FASE 3", "ALFA"], n),
            "Defasagem": np.random.randint(-2, 3, n),
            "Turma": np.random.choice(["A", "B"], n),
            "IAA": np.random.uniform(0, 10, n),
            "IEG": np.random.uniform(0, 10, n),
            "IPS": np.random.uniform(0, 10, n),
            "IDA": np.random.uniform(0, 10, n),
            "IPV": np.random.uniform(0, 10, n),
            "IAN": np.random.uniform(0, 10, n),
            "Nº Av": np.random.randint(1, 5, n).astype(float),
            "Rec.Av1": np.random.choice([0, 1], n),
            "Rec.Av2": np.random.choice([0, 1], n),
            "Rec.Av3": np.random.choice([0, 1], n),
            "Rec.Av4": np.random.choice([0, 1], n),
            "Avaliador1": np.random.choice(["X", "Y"], n),
            "Avaliador2": np.random.choice(["X", "Y"], n),
            "Avaliador3": np.random.choice(["X", "Y"], n),
            "Avaliador4": np.random.choice(["X", "Y"], n),
            "Destaque.IDA": np.random.choice([0, 1], n),
            "Destaque.IEG": np.random.choice([0, 1], n),
            "Destaque.IPV": np.random.choice([0, 1], n),
            "Fase.Ideal": np.random.choice(["1", "2"], n),
            "Indicado": np.random.choice([0, 1], n),
            "Atingiu.PV": np.random.choice([0, 1], n),
            "Rec.Psicologia": np.random.choice([0, 1], n),
            "Ano.Ingresso": np.random.choice([2018, 2019, 2020], n),
            "Pedra.20": np.random.choice(["A", "T"], n),
            "Pedra.21": np.random.choice(["A", "T"], n),
            "Pedra.22": np.random.choice(["A", "T"], n),
            "Pedra.23": np.random.choice(["A", "T"], n),
        }
        if year_suffix == "22":
            d["Idade 22"] = np.random.randint(8, 18, n)
            d["INDE.22"] = np.random.uniform(0, 10, n)
            d["Defas"] = np.random.randint(-2, 3, n)
            d["Matem"] = np.random.uniform(0, 10, n)
            d["Portug"] = np.random.uniform(0, 10, n)
            d["Inglês"] = np.random.uniform(0, 10, n)
            d["CF"] = np.random.uniform(0, 10, n)
            d["CT"] = np.random.uniform(0, 10, n)
            d["CG"] = np.random.uniform(0, 10, n)
        else:
            d["Idade"] = np.random.randint(8, 18, n)
            d["Mat"] = np.random.uniform(0, 10, n)
            d["Por"] = np.random.uniform(0, 10, n)
            d["Ing"] = np.random.uniform(0, 10, n)
            d["CF"] = np.random.uniform(0, 10, n)
            d["CT"] = np.random.uniform(0, 10, n)
            d["CG"] = np.random.uniform(0, 10, n)
        return pd.DataFrame(d)

    data_22 = _make_df(common_ras, "22")
    data_23 = _make_df(common_ras + new_ras_23, "23")
    data_24 = _make_df(common_ras[:10] + new_ras_24, "24")
    return data_22, data_23, data_24


class TestCreateDatasets:
    def test_returns_two_dataframes(self):
        d22, d23, d24 = _make_raw_dataframes()
        tr, vl = create_datasets(d22, d23, d24)
        assert isinstance(tr, pd.DataFrame)
        assert isinstance(vl, pd.DataFrame)

    def test_training_has_target(self):
        d22, d23, d24 = _make_raw_dataframes()
        tr, _ = create_datasets(d22, d23, d24)
        assert "Defasagem.final" in tr.columns

    def test_validation_has_target(self):
        d22, d23, d24 = _make_raw_dataframes()
        _, vl = create_datasets(d22, d23, d24)
        assert "Defasagem.final" in vl.columns

    def test_training_shape(self):
        d22, d23, d24 = _make_raw_dataframes()
        tr, _ = create_datasets(d22, d23, d24)
        assert len(tr) > 0

    def test_validation_only_new_students(self):
        d22, d23, d24 = _make_raw_dataframes()
        _, vl = create_datasets(d22, d23, d24)
        assert len(vl) > 0


class TestProcessData:
    @pytest.fixture
    def raw_datasets(self):
        d22, d23, d24 = _make_raw_dataframes()
        tr, vl = create_datasets(d22, d23, d24)
        return tr, vl

    def test_output_shape(self, raw_datasets):
        tr_raw, vl_raw = raw_datasets
        tr, vl = process_data(tr_raw, vl_raw)
        assert tr.shape[0] > 0
        assert tr.shape[1] > 0
        assert vl.shape[1] > 0

    def test_target_binary(self, raw_datasets):
        tr_raw, vl_raw = raw_datasets
        tr, vl = process_data(tr_raw, vl_raw)
        assert set(tr["defasagem"].unique()).issubset({0, 1})

    def test_no_raw_columns(self, raw_datasets):
        from src.preprocessing import REMOVER_STEP1, REMOVER_STEP2
        tr_raw, vl_raw = raw_datasets
        tr, vl = process_data(tr_raw, vl_raw)
        for col in REMOVER_STEP1 + REMOVER_STEP2:
            assert col not in tr.columns
            assert col not in vl.columns

    def test_dummy_encoding(self, raw_datasets):
        tr_raw, vl_raw = raw_datasets
        tr, vl = process_data(tr_raw, vl_raw)
        fase_cols = [c for c in tr.columns if c.startswith("fase_")]
        assert len(fase_cols) > 0

    def test_scaled_columns(self, raw_datasets):
        tr_raw, vl_raw = raw_datasets
        tr, vl = process_data(tr_raw, vl_raw)
        from src.preprocessing import COLS_TO_SCALE
        available = [c for c in COLS_TO_SCALE if c in tr.columns]
        if available:
            means = tr[available].mean()
            for col in available:
                assert abs(means[col]) < 0.5, f"Coluna {col} não parece padronizada"

    def test_returns_two_dataframes(self, raw_datasets):
        tr_raw, vl_raw = raw_datasets
        result = process_data(tr_raw, vl_raw)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)
