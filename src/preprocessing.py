"""
Carregamento, limpeza e preparação dos dados.
Conversão fiel do código R (Descriptive.R).
"""
import unicodedata
import numpy as np
import pandas as pd


# === Constantes ===

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


def _normalize_colname(name: str) -> str:
    """Lower case, sem acento, espaço vira ponto."""
    name = str(name).lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.replace(" ", ".")


def _encode_instituicao(val) -> int:
    if pd.isna(val):
        return np.nan
    v = unicodedata.normalize("NFD", str(val).lower())
    v = "".join(c for c in v if unicodedata.category(c) != "Mn")
    return 1 if "publica" in v else 0


def load_sheets(filepath: str) -> tuple:
    """Carrega as 3 abas do Excel (PEDE2022, PEDE2023, PEDE2024)."""
    data_22 = pd.read_excel(filepath, sheet_name="PEDE2022")
    data_23 = pd.read_excel(filepath, sheet_name="PEDE2023")
    data_24 = pd.read_excel(filepath, sheet_name="PEDE2024")
    print(f"  2022: {data_22.shape} | 2023: {data_23.shape} | 2024: {data_24.shape}")
    return data_22, data_23, data_24


def create_datasets(data_22, data_23, data_24) -> tuple:
    """Cria training set (2022→2023) e validation set (2023→2024).

    Training: alunos presentes em 2022 que permaneceram em 2023.
    Validation: alunos novos em 2023 (ausentes em 2022) que permaneceram em 2024.
    """
    # Training
    training_students = data_22["RA"][data_22["RA"].isin(data_23["RA"])]
    data_22_tr = data_22[data_22["RA"].isin(training_students)].copy()
    data_23_tr = data_23[data_23["RA"].isin(training_students)].copy()

    # Validation
    validation_students = data_23["RA"][~data_23["RA"].isin(data_22["RA"])]
    data_24_vl = data_24[data_24["RA"].isin(validation_students)].copy()
    data_23_vl = data_23[data_23["RA"].isin(data_24_vl["RA"])].copy()

    # Merge: defasagem do ano seguinte como coluna target
    defas_23 = data_23_tr[["RA", "Defasagem"]].rename(columns={"Defasagem": "Defasagem.final"})
    defas_24 = data_24_vl[["RA", "Defasagem"]].rename(columns={"Defasagem": "Defasagem.final"})

    training_set = data_22_tr.merge(defas_23, on="RA", how="left")
    validation_set = data_23_vl.merge(defas_24, on="RA", how="left")

    print(f"  Training: {training_set.shape} | Validation: {validation_set.shape}")
    return training_set, validation_set


def process_data(training_set: pd.DataFrame, validation_set: pd.DataFrame) -> tuple:
    """Pipeline completa de pré-processamento (equivalente ao R).

    Etapas:
    1. Normalizar nomes de colunas
    2. Alinhar colunas entre datasets
    3. Criar target binário
    4. Remover features pouco informativas
    5. Tratar coluna fase
    6. Hot encoding (instituição, gênero)
    7. Remover colunas com muitos NAs
    8. Dummy encoding de fase
    9. Reordenar e padronizar (z-score)
    """
    tr = training_set.copy()
    vl = validation_set.copy()

    # --- 1. Normalizar nomes ---
    tr.columns = [_normalize_colname(c) for c in tr.columns]
    vl.columns = [_normalize_colname(c) for c in vl.columns]

    # --- 2. Renomear para consistência ---
    tr = tr.rename(columns={"defas": "defasagem.inicial", "idade.22": "idade",
                            "portug": "protugues", "matem": "matematica"})
    vl = vl.rename(columns={"defasagem": "defasagem.inicial", "por": "portugues",
                            "ing": "ingles", "mat": "matematica"})

    # Remover colunas exclusivas
    common = set(tr.columns) & set(vl.columns)
    tr = tr.drop(columns=[c for c in tr.columns if c not in common], errors="ignore")
    vl = vl.drop(columns=[c for c in vl.columns if c not in common], errors="ignore")

    # --- 3. Target binário ---
    tr["defasagem.diferenca"] = tr["defasagem.final"] - tr["defasagem.inicial"]
    vl["defasagem.diferenca"] = vl["defasagem.final"] - vl["defasagem.inicial"]
    tr["defasagem"] = (tr["defasagem.diferenca"] < 0).astype(int)
    vl["defasagem"] = (vl["defasagem.diferenca"] < 0).astype(int)

    print(f"  Target training: {tr['defasagem'].value_counts().to_dict()}")
    print(f"  Target validation: {vl['defasagem'].value_counts().to_dict()}")

    # --- 4. Remover features pouco informativas ---
    tr = tr.drop(columns=[c for c in REMOVER_STEP1 if c in tr.columns], errors="ignore")
    vl = vl.drop(columns=[c for c in REMOVER_STEP1 if c in vl.columns], errors="ignore")

    # --- 5. Tratar coluna fase ---
    vl["fase"] = vl["fase"].replace("ALFA", "0")
    vl["fase"] = vl["fase"].astype(str).str.replace("FASE", "", regex=False).str.strip()
    for col in ["fase"] + COLS_TO_SCALE:
        if col in tr.columns:
            tr[col] = pd.to_numeric(tr[col], errors="coerce")
        if col in vl.columns:
            vl[col] = pd.to_numeric(vl[col], errors="coerce")

    # --- 6. Hot encoding ---
    tr["instituicao.publica"] = tr["instituicao.de.ensino"].apply(_encode_instituicao)
    vl["instituicao.publica"] = vl["instituicao.de.ensino"].apply(_encode_instituicao)

    tr["genero"] = tr["genero"].str.lower()
    vl["genero"] = vl["genero"].str.lower()
    tr["genero.masculino"] = (tr["genero"] == "menino").astype(int)
    vl["genero.masculino"] = (vl["genero"] == "masculino").astype(int)

    # --- 7. Remover colunas NAs ---
    tr = tr.drop(columns=[c for c in REMOVER_STEP2 if c in tr.columns], errors="ignore")
    vl = vl.drop(columns=[c for c in REMOVER_STEP2 if c in vl.columns], errors="ignore")

    na_pct = vl.isnull().mean()
    high_na = na_pct[na_pct > 0.5].index.tolist()
    if high_na:
        tr = tr.drop(columns=[c for c in high_na if c in tr.columns])
        vl = vl.drop(columns=[c for c in high_na if c in vl.columns])

    vl = vl.dropna()

    # --- 8. Dummy encoding de fase ---
    tr = pd.get_dummies(tr, columns=["fase"], prefix="fase", drop_first=False, dtype=int)
    vl = pd.get_dummies(vl, columns=["fase"], prefix="fase", drop_first=False, dtype=int)

    all_fase = sorted(
        set(c for c in tr.columns if c.startswith("fase_"))
        | set(c for c in vl.columns if c.startswith("fase_"))
    )
    for col in all_fase:
        if col not in tr.columns:
            tr[col] = 0
        if col not in vl.columns:
            vl[col] = 0

    tr.columns = [c.replace(" ", "") for c in tr.columns]
    vl.columns = [c.replace(" ", "") for c in vl.columns]

    # --- 9. Reordenar colunas ---
    fase_cols = sorted(c for c in tr.columns if c.startswith("fase_"))
    reorder = (
        ["ra", "idade", "genero.masculino", "instituicao.publica",
         "nº.av", "iaa", "ieg", "ips", "ida", "ipv", "ian", "matematica"]
        + fase_cols
        + ["defasagem"]
    )
    reorder = [c for c in reorder if c in tr.columns]
    tr = tr[reorder]
    vl = vl[[c for c in reorder if c in vl.columns]]

    # --- 10. Z-score ---
    from sklearn.preprocessing import StandardScaler

    available_scale = [c for c in COLS_TO_SCALE if c in tr.columns]
    scaler = StandardScaler()
    tr[available_scale] = scaler.fit_transform(tr[available_scale])
    vl[available_scale] = scaler.transform(vl[available_scale])

    # RA como index
    tr = tr.set_index("ra")
    vl = vl.set_index("ra")

    print(f"  Final -> Training: {tr.shape} | Validation: {vl.shape}")
    print(f"  Colunas: {list(tr.columns)}")
    return tr, vl
