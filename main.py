"""
Pipeline principal — orquestra todas as etapas.

Uso:
    python main.py
    python main.py --data caminho/para/arquivo.xlsx
"""
import os
import sys
import warnings
import argparse

import pandas as pd

warnings.filterwarnings("ignore")

# Garante que imports funcionem de qualquer diretório
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_sheets, create_datasets, process_data
from src.train import (
    split_data, prepare_validation, train_all_models,
    save_models, evaluate_all,
)
from src.visualize import plot_comparison, plot_roc_curves, plot_confusion_matrices, plot_consolidated


def main(data_path: str):
    OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(OUTPUT, exist_ok=True)

    print("=" * 70)
    print("  PASSOS MÁGICOS — PIPELINE DE MACHINE LEARNING")
    print("=" * 70)

    # =================================================================
    # PARTE 1: PRÉ-PROCESSAMENTO (conversão do código R)
    # =================================================================

    print("\n[1] Importando datasets...")
    data_22, data_23, data_24 = load_sheets(data_path)

    print("\n[2] Criando training e validation sets...")
    training_set, validation_set = create_datasets(data_22, data_23, data_24)

    print("\n[3] Pré-processamento...")
    training_set, validation_set = process_data(training_set, validation_set)

    training_set.to_csv(os.path.join(OUTPUT, "training_set.csv"))
    validation_set.to_csv(os.path.join(OUTPUT, "validation_set.csv"))
    print(f"  Dados salvos em {OUTPUT}/training_set.csv e validation_set.csv")

    # =================================================================
    # PARTE 2: MODELAGEM (11 etapas)
    # =================================================================

    print("\n" + "=" * 70)
    print("  MODELAGEM")
    print("=" * 70)

    # Etapa 1: Split 70/30
    print("\n[Etapa 1] Split 70/30...")
    X_train, X_test, y_train, y_test = split_data(training_set)
    X_val, y_val = prepare_validation(validation_set, X_train)

    # Etapas 2-5: Grid Search + CV
    print("\n[Etapas 2-5] Grid Search com 5-fold CV...")
    grid_results = train_all_models(X_train, y_train)

    # Etapa 6: Salvar modelos
    print("\n[Etapa 6] Salvando modelos...")
    models = save_models(grid_results, MODELS)

    # Etapa 7: Métricas e gráficos — Treino
    print("\n[Etapa 7] Avaliação nos dados de TREINO...")
    train_df = evaluate_all(models, X_train, y_train, "Treino")
    plot_comparison(train_df, "Dados de Treino",
                    os.path.join(OUTPUT, "etapa7_comparacao_treino.png"))
    plot_roc_curves(models, X_train, y_train, "Dados de Treino",
                    os.path.join(OUTPUT, "etapa7_roc_treino.png"))
    plot_confusion_matrices(models, X_train, y_train, "Dados de Treino",
                            os.path.join(OUTPUT, "etapa7_confusion_treino.png"))

    # Etapas 8-9: Aplicar no teste + gráficos
    print("\n[Etapas 8-9] Avaliação nos dados de TESTE...")
    test_df = evaluate_all(models, X_test, y_test, "Teste")
    plot_comparison(test_df, "Dados de Teste",
                    os.path.join(OUTPUT, "etapa9_comparacao_teste.png"))
    plot_roc_curves(models, X_test, y_test, "Dados de Teste",
                    os.path.join(OUTPUT, "etapa9_roc_teste.png"))
    plot_confusion_matrices(models, X_test, y_test, "Dados de Teste",
                            os.path.join(OUTPUT, "etapa9_confusion_teste.png"))

    # Etapas 10-11: Aplicar na validação + gráficos
    print("\n[Etapas 10-11] Avaliação nos dados de VALIDAÇÃO...")
    val_df = evaluate_all(models, X_val, y_val, "Validação")
    plot_comparison(val_df, "Dados de Validação",
                    os.path.join(OUTPUT, "etapa11_comparacao_validacao.png"))
    plot_roc_curves(models, X_val, y_val, "Dados de Validação",
                    os.path.join(OUTPUT, "etapa11_roc_validacao.png"))
    plot_confusion_matrices(models, X_val, y_val, "Dados de Validação",
                            os.path.join(OUTPUT, "etapa11_confusion_validacao.png"))

    # =================================================================
    # RESUMO CONSOLIDADO
    # =================================================================

    print("\n" + "=" * 70)
    print("  RESUMO")
    print("=" * 70)

    summary = pd.concat([train_df, test_df, val_df], ignore_index=True)
    summary.to_csv(os.path.join(OUTPUT, "resultados_completos.csv"), index=False)
    plot_consolidated(summary, os.path.join(OUTPUT, "resumo_consolidado.png"))

    print("\n" + summary[["Model", "Accuracy", "Sensitivity", "Specificity", "AUC", "Etapa"]]
          .to_string(index=False))

    print("\n" + "=" * 70)
    print("  PIPELINE CONCLUÍDA!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/BASE_DE_DADOS_PEDE.xlsx",
                        help="Caminho para o arquivo Excel com os dados")
    args = parser.parse_args()
    main(args.data)
