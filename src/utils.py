"""
Funções auxiliares e constantes do projeto.
"""
import logging
import os
import pickle
import json
import unicodedata
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "app" / "model"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

COLS_TO_SCALE = ["idade", "nº.av", "iaa", "ieg", "ips", "ida", "ipv", "ian", "matematica"]

REMOVER_STEP1 = [
    "avaliador1", "avaliador2", "avaliador3", "avaliador4",
    "rec.av1", "rec.av2", "rec.av3", "rec.av4",
    "turma", "destaque.ida", "destaque.ieg",
    "fase.ideal", "indicado", "atingiu.pv", "inde.22",
]

REMOVER_STEP2 = [
    "genero", "instituicao.de.ensino", "rec.psicologia",
    "pedra.20", "pedra.21", "pedra.22", "pedra.23",
    "ingles", "cf", "ct", "cg",
    "defasagem.inicial", "defasagem.final", "defasagem.diferenca",
    "ano.ingresso", "destaque.ipv",
]


def setup_logger(name, level=logging.INFO):
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def normalize_colname(name):
    name = str(name).lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.replace(" ", ".")
    return name


def save_model(model, filename="model.pkl"):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODEL_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return str(filepath)


def load_model(filename="model.pkl"):
    filepath = MODEL_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(data, filename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODEL_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return str(filepath)
