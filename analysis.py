"""
Análise exploratória (EDA) + Feature Importance.

Gera gráficos descritivos e de importância de features a partir dos dados
preprocessados e modelos já treinados, sem re-executar o pipeline de treino.

Uso:
    python analysis.py
"""
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_sheets
from src.utils import load_model, OUTPUT_DIR
from src.visualize import (
    plot_missing_data,
    plot_target_distribution,
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_feature_importance,
)

DATA_PATH = "data/BASE_DE_DADOS_PEDE.xlsx"

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Neural Network": "neural_network.pkl",
}


def main():
    output = str(OUTPUT_DIR)
    os.makedirs(output, exist_ok=True)

    print("=" * 70)
    print("  ANÁLISE EXPLORATÓRIA + FEATURE IMPORTANCE")
    print("=" * 70)

    # 1. Dados brutos (para gráfico de missing data)
    print("\n[1] Carregando dados brutos do Excel...")
    data_22, data_23, data_24 = load_sheets(DATA_PATH)
    sheets = {"PEDE2022": data_22, "PEDE2023": data_23, "PEDE2024": data_24}

    # 2. Dados preprocessados
    print("\n[2] Carregando training set preprocessado...")
    df = pd.read_csv(os.path.join(output, "training_set.csv"), index_col="ra")
    print(f"  Shape: {df.shape}")

    # 3. Modelos treinados
    print("\n[3] Carregando modelos...")
    models = {}
    for name, filename in MODEL_FILES.items():
        models[name] = load_model(filename)
        print(f"  {name}: OK")

    # 4. Separar features e target
    X = df.drop(columns=["defasagem"])
    y = df["defasagem"]
    feature_names = list(X.columns)

    # 5. Gerar gráficos
    print("\n[4] Gerando gráficos...")

    plot_missing_data(sheets, os.path.join(output, "descritivo_missing.png"))
    plot_target_distribution(df, os.path.join(output, "descritivo_target.png"))
    plot_feature_distributions(df, os.path.join(output, "descritivo_features.png"))
    plot_correlation_heatmap(df, os.path.join(output, "descritivo_correlacao.png"))
    plot_feature_importance(models, feature_names, X, y,
                            os.path.join(output, "feature_importance.png"))

    print("\n" + "=" * 70)
    print("  ANÁLISE CONCLUÍDA!")
    print(f"  Gráficos salvos em {output}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
