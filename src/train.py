"""
Treinamento, tuning e avaliação dos modelos.
Implementa as 11 etapas definidas no Descriptive.R.
"""
import pickle
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.utils import resample


# =========================================================================
# ETAPA 1: SPLIT 70/30
# =========================================================================

def split_data(training_set: pd.DataFrame, test_size: float = 0.3):
    """Divide training_set em 70% treino / 30% teste (estratificado)."""
    X = training_set.drop(columns=["defasagem"])
    y = training_set["defasagem"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    print(f"  X_train: {X_train.shape} | y_train: {y_train.value_counts().to_dict()}")
    print(f"  X_test:  {X_test.shape}  | y_test:  {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def prepare_validation(validation_set: pd.DataFrame, X_train: pd.DataFrame):
    """Prepara validation set alinhando colunas com training."""
    X_val = validation_set.drop(columns=["defasagem"])
    y_val = validation_set["defasagem"]

    for c in set(X_train.columns) - set(X_val.columns):
        X_val[c] = 0
    X_val = X_val[X_train.columns]

    print(f"  X_val:   {X_val.shape}   | y_val:   {y_val.value_counts().to_dict()}")
    return X_val, y_val


# =========================================================================
# ETAPAS 2-5: GRID SEARCH COM 5-FOLD CV
# =========================================================================

def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Treina 4 modelos com GridSearchCV + 5-fold CV.

    Modelos: Random Forest, Logistic Regression, Gradient Boosting, Neural Network.
    Retorna dict com nome → best_estimator.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Tratar NaN
    X_tr = X_train.fillna(X_train.median())

    results = {}

    # --- Random Forest ---
    print("\n  [1/4] Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
        {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 8, None],
         "min_samples_leaf": [1, 3, 5]},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    rf_grid.fit(X_tr, y_train)
    results["Random Forest"] = rf_grid
    print(f"         Best AUC (CV): {rf_grid.best_score_:.4f} | Params: {rf_grid.best_params_}")

    # --- Logistic Regression ---
    print("\n  [2/4] Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, class_weight="balanced"),
        {"C": [0.001, 0.01, 0.1, 1, 10], "penalty": ["l1", "l2"],
         "solver": ["saga"], "max_iter": [5000]},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    lr_grid.fit(X_tr, y_train)
    results["Logistic Regression"] = lr_grid
    print(f"         Best AUC (CV): {lr_grid.best_score_:.4f} | Params: {lr_grid.best_params_}")

    # --- Gradient Boosting ---
    print("\n  [3/4] Gradient Boosting...")
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 5],
         "learning_rate": [0.01, 0.05, 0.1], "min_samples_leaf": [3, 5, 10]},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    gb_grid.fit(X_tr, y_train)
    results["Gradient Boosting"] = gb_grid
    print(f"         Best AUC (CV): {gb_grid.best_score_:.4f} | Params: {gb_grid.best_params_}")

    # --- Neural Network ---
    print("\n  [4/4] Neural Network (MLP)...")
    # Oversample minority class for MLP (não suporta sample_weight)
    minority_mask = y_train == 1
    X_majority, y_majority = X_tr[~minority_mask], y_train[~minority_mask]
    X_minority, y_minority = X_tr[minority_mask], y_train[minority_mask]
    X_min_up, y_min_up = resample(
        X_minority, y_minority,
        replace=True, n_samples=len(y_majority), random_state=42,
    )
    X_balanced = pd.concat([X_majority, X_min_up])
    y_balanced = pd.concat([y_majority, y_min_up])

    nn_grid = GridSearchCV(
        MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.15),
        {"hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16)],
         "alpha": [0.0001, 0.001, 0.01], "learning_rate_init": [0.001, 0.01],
         "max_iter": [1000]},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    nn_grid.fit(X_balanced, y_balanced)
    results["Neural Network"] = nn_grid
    print(f"         Best AUC (CV): {nn_grid.best_score_:.4f} | Params: {nn_grid.best_params_}")

    return results


# =========================================================================
# ETAPA 6: SALVAR MODELOS
# =========================================================================

def save_models(grid_results: dict, model_dir: str) -> dict:
    """Salva modelos ótimos (.pkl) e parâmetros (JSON)."""
    os.makedirs(model_dir, exist_ok=True)
    models = {}

    for name, grid in grid_results.items():
        models[name] = grid.best_estimator_
        fpath = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.pkl")
        with open(fpath, "wb") as f:
            pickle.dump(grid.best_estimator_, f)
        print(f"  Salvo: {fpath}")

    params = {name: grid.best_params_ for name, grid in grid_results.items()}
    params_path = os.path.join(model_dir, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2, default=str)
    print(f"  Parâmetros: {params_path}")

    return models


# =========================================================================
# ETAPAS 7-11: AVALIAÇÃO E MÉTRICAS
# =========================================================================

def compute_metrics(model, X, y, model_name: str) -> dict:
    """Calcula Accuracy, Sensitivity, Specificity e AUC."""
    X_clean = X.fillna(X.median())
    y_pred = model.predict(X_clean)
    y_proba = model.predict_proba(X_clean)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Sensitivity": recall_score(y, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "AUC": roc_auc_score(y, y_proba),
        "F1": f1_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
    }


def evaluate_all(models: dict, X, y, stage_name: str) -> pd.DataFrame:
    """Avalia todos os modelos em um dataset e retorna DataFrame com métricas."""
    print(f"\n  === {stage_name} ===")
    rows = []
    for name, model in models.items():
        m = compute_metrics(model, X, y, name)
        rows.append(m)
        print(f"  {name}: Acc={m['Accuracy']:.4f} | Sens={m['Sensitivity']:.4f} "
              f"| Spec={m['Specificity']:.4f} | AUC={m['AUC']:.4f}")

    df = pd.DataFrame(rows)
    df["Etapa"] = stage_name
    return df
