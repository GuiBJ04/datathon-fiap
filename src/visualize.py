"""
Geração de gráficos de comparação entre modelos.
Etapas 7, 9 e 11 do pipeline.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
import seaborn as sns
import numpy as np


COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]


def plot_comparison(results_df: pd.DataFrame, stage_name: str, filepath: str):
    """Gera gráfico de barras comparando Accuracy, Sensitivity, Specificity e AUC."""
    metrics = ["Accuracy", "Sensitivity", "Specificity", "AUC"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Comparação de Modelos — {stage_name}", fontsize=18, fontweight="bold", y=1.02)

    for idx, (metric, color) in enumerate(zip(metrics, COLORS)):
        ax = axes[idx // 2][idx % 2]
        bars = ax.bar(results_df["Model"], results_df[metric],
                      color=color, edgecolor="black", alpha=0.85)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax.set_xticklabels(results_df["Model"], rotation=25, ha="right", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico: {filepath}")


def plot_roc_curves(models: dict, X, y, stage_name: str, filepath: str):
    """Plota curvas ROC de todos os modelos."""
    X_clean = X.fillna(X.median())

    fig, ax = plt.subplots(figsize=(10, 8))
    for (name, model), color in zip(models.items(), COLORS):
        y_proba = model.predict_proba(X_clean)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc_val = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"Curvas ROC — {stage_name}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curvas ROC: {filepath}")


def plot_confusion_matrices(models: dict, X, y, stage_name: str, filepath: str):
    """Plota matrizes de confusão de todos os modelos em um único arquivo."""
    X_clean = X.fillna(X.median())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle(f"Matrizes de Confusão — {stage_name}", fontsize=16, fontweight="bold", y=1.02)

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_clean)
        cm = confusion_matrix(y, y_pred)

        ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predito", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold", color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matrizes de confusão: {filepath}")


def plot_consolidated(summary: pd.DataFrame, filepath: str):
    """Gráfico consolidado: Treino vs Teste vs Validação para todos os modelos."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle("Comparação Consolidada — Treino vs Teste vs Validação",
                 fontsize=18, fontweight="bold")

    for idx, metric in enumerate(["Accuracy", "Sensitivity", "Specificity", "AUC"]):
        ax = axes[idx]
        pivot = summary.pivot(index="Model", columns="Etapa", values=metric)
        pivot = pivot[["Treino", "Teste", "Validação"]]
        pivot.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85, width=0.7)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Consolidado: {filepath}")


# =====================================================================
# Análise Exploratória (EDA) + Feature Importance
# =====================================================================


def plot_missing_data(sheets: dict, filepath: str):
    """Gráfico de dados faltantes por coluna nos dados brutos (3 subplots, 1 por ano)."""
    years = list(sheets.keys())
    fig, axes = plt.subplots(1, len(years), figsize=(8 * len(years), 10))
    if len(years) == 1:
        axes = [axes]

    fig.suptitle("Dados Faltantes por Coluna (Dados Brutos)", fontsize=18, fontweight="bold")

    for ax, year in zip(axes, years):
        df = sheets[year]
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)

        if missing.empty:
            ax.set_title(f"{year}\n(sem dados faltantes)", fontsize=14, fontweight="bold")
            ax.set_yticks([])
            continue

        pct = (missing / len(df) * 100).round(1)
        bars = ax.barh(range(len(missing)), missing.values, color=COLORS[0], edgecolor="black", alpha=0.85)
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index, fontsize=9)
        ax.set_xlabel("Contagem de NaN", fontsize=11)
        ax.set_title(f"{year}", fontsize=14, fontweight="bold")

        for i, (bar, count, p) in enumerate(zip(bars, missing.values, pct.values)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{count} ({p}%)", va="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico: {filepath}")


def plot_target_distribution(df: pd.DataFrame, filepath: str):
    """Gráfico de barras da distribuição da variável target (defasagem)."""
    counts = df["defasagem"].value_counts().sort_index()
    total = len(df)
    labels = [f"Classe {c}" for c in counts.index]
    pcts = (counts / total * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts.values, color=[COLORS[0], COLORS[1]], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Contagem", fontsize=13)
    ax.set_title("Distribuição da Variável Target (Defasagem)", fontsize=16, fontweight="bold")

    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{count} ({pct}%)", ha="center", va="bottom", fontweight="bold", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico: {filepath}")


def plot_feature_distributions(df: pd.DataFrame, filepath: str):
    """Grid de histogramas para as features numéricas, separados por classe."""
    from src.utils import COLS_TO_SCALE

    features = [c for c in COLS_TO_SCALE if c in df.columns]
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle("Distribuição das Features Numéricas por Classe", fontsize=18, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        for cls, color, label in [(0, COLORS[0], "Classe 0"), (1, COLORS[1], "Classe 1")]:
            subset = df[df["defasagem"] == cls][feat].dropna()
            ax.hist(subset, bins=20, color=color, alpha=0.6, label=label, edgecolor="black")
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=9)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico: {filepath}")


def plot_correlation_heatmap(df: pd.DataFrame, filepath: str):
    """Heatmap de correlação entre features numéricas e target."""
    from src.utils import COLS_TO_SCALE

    cols = [c for c in COLS_TO_SCALE if c in df.columns] + ["defasagem"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, annot_kws={"fontsize": 10})
    ax.set_title("Heatmap de Correlação", fontsize=16, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico: {filepath}")


def plot_feature_importance(models: dict, feature_names: list, X, y, filepath: str):
    """Grid 2x2 com importância de features por modelo."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Feature Importance por Modelo", fontsize=18, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, ((name, model), color) in enumerate(zip(models.items(), COLORS)):
        ax = axes_flat[idx]

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
            importance = importance / importance.sum()
        else:
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importance = result.importances_mean

        sorted_idx = np.argsort(importance)
        ax.barh(range(len(sorted_idx)), importance[sorted_idx],
                color=color, edgecolor="black", alpha=0.85)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel("Importância", fontsize=11)
        ax.set_title(name, fontsize=14, fontweight="bold")

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico: {filepath}")
