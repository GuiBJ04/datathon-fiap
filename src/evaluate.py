"""
Avaliação e gráficos de comparação (Etapas 7 a 11).
"""
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.utils import OUTPUT_DIR, setup_logger

logger = setup_logger("evaluate")


def compute_metrics(model, X, y, model_name):
    """Calcula accuracy, sensitivity, specificity e AUC."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
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


def evaluate_all(models, X, y, stage_name):
    """Calcula métricas de todos os modelos em um dataset."""
    results = []
    for name, model in models.items():
        m = compute_metrics(model, X, y, name)
        results.append(m)
        logger.info(f"  {name} ({stage_name}): Acc={m['Accuracy']:.3f} Sens={m['Sensitivity']:.3f} Spec={m['Specificity']:.3f} AUC={m['AUC']:.3f}")
    return pd.DataFrame(results)


def plot_comparison(results_df, stage_name, filename):
    """Gráficos de barras comparando métricas entre modelos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Sensitivity", "Specificity", "AUC"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Comparação de Modelos — {stage_name}", fontsize=18, fontweight="bold", y=1.02)

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2][idx % 2]
        bars = ax.bar(results_df["Model"], results_df[metric], color=color, edgecolor="black", alpha=0.85)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.set_xticklabels(results_df["Model"], rotation=25, ha="right", fontsize=10)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Gráfico salvo: {filepath}")


def plot_roc_curves(models, X, y, stage_name, filename):
    """Curvas ROC de todos os modelos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc_val = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"Curvas ROC — {stage_name}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ROC salvo: {filepath}")


def plot_consolidated(train_df, test_df, val_df, filename="resumo_consolidado.png"):
    """Gráfico consolidado Treino × Teste × Validação."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame()
    for stage, df in [("Treino", train_df), ("Teste", test_df), ("Validação", val_df)]:
        temp = df[["Model", "Accuracy", "Sensitivity", "Specificity", "AUC"]].copy()
        temp["Etapa"] = stage
        summary = pd.concat([summary, temp], ignore_index=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle("Comparação Consolidada — Treino vs Teste vs Validação", fontsize=18, fontweight="bold")

    for idx, metric in enumerate(["Accuracy", "Sensitivity", "Specificity", "AUC"]):
        ax = axes[idx]
        pivot = summary.pivot(index="Model", columns="Etapa", values=metric)
        pivot = pivot[["Treino", "Teste", "Validação"]]
        pivot.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85, width=0.7)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    summary.to_csv(OUTPUT_DIR / "resultados_completos.csv", index=False)
    logger.info(f"  Consolidado: {filepath}")
    return summary
