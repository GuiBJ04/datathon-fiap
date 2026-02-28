"""
=============================================================================
Passos Mágicos — Pipeline completa (ponto de entrada)
Conversão do Descriptive.R + 11 etapas de modelagem
=============================================================================

Uso:
    python pipeline.py                           # usa dados em data/
    python pipeline.py caminho/para/dados.xlsx   # caminho customizado
"""
import sys
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing import run_preprocessing
from src.train import split_data, prepare_validation, train_all_models, save_all_models
from src.evaluate import (
    evaluate_all, plot_comparison, plot_roc_curves, plot_consolidated,
)

def main(data_path="data/BASE_DE_DADOS_PEDE__-_DATATHON.xlsx"):

    print("=" * 70)
    print("  PASSOS MÁGICOS — PIPELINE DE MACHINE LEARNING")
    print("=" * 70)

    # ── Pré-processamento (conversão do R) ────────────────────────────
    print("\n[PRÉ-PROCESSAMENTO] Convertendo código R → Python...")
    training_set, validation_set = run_preprocessing(data_path)

    print(f"\n  Training set: {training_set.shape}")
    print(f"  Validation set: {validation_set.shape}")
    print(f"  Training target: {training_set['defasagem'].value_counts().to_dict()}")
    print(f"  Validation target: {validation_set['defasagem'].value_counts().to_dict()}")

    # ── Etapa 1: Split 70/30 ─────────────────────────────────────────
    print("\n[ETAPA 1] Split training.set em 70% treino / 30% teste...")
    X_train, X_test, y_train, y_test = split_data(training_set)
    X_val, y_val = prepare_validation(validation_set, X_train)

    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape} | X_val: {X_val.shape}")

    # ── Etapas 2-5: Treino com Grid Search + CV ──────────────────────
    print("\n[ETAPAS 2-5] Grid Search com 5-fold CV...")
    models, best_params = train_all_models(X_train, y_train)

    # ── Etapa 6: Salvar modelos ──────────────────────────────────────
    print("\n[ETAPA 6] Salvando modelos e parâmetros...")
    save_all_models(models, best_params)

    # ── Etapa 7: Métricas e gráficos de treino ───────────────────────
    print("\n[ETAPA 7] Métricas nos dados de TREINO...")
    train_df = evaluate_all(models, X_train, y_train, "Treino")
    plot_comparison(train_df, "Dados de Treino", "etapa7_comparacao_treino.png")
    plot_roc_curves(models, X_train, y_train, "Dados de Treino", "etapa7_roc_treino.png")

    # ── Etapa 8-9: Métricas e gráficos de teste ─────────────────────
    print("\n[ETAPAS 8-9] Métricas e gráficos nos dados de TESTE...")
    test_df = evaluate_all(models, X_test, y_test, "Teste")
    plot_comparison(test_df, "Dados de Teste", "etapa9_comparacao_teste.png")
    plot_roc_curves(models, X_test, y_test, "Dados de Teste", "etapa9_roc_teste.png")

    # ── Etapa 10-11: Métricas e gráficos de validação ────────────────
    print("\n[ETAPAS 10-11] Métricas e gráficos nos dados de VALIDAÇÃO...")
    val_df = evaluate_all(models, X_val, y_val, "Validação")
    plot_comparison(val_df, "Dados de Validação", "etapa11_comparacao_validacao.png")
    plot_roc_curves(models, X_val, y_val, "Dados de Validação", "etapa11_roc_validacao.png")

    # ── Resumo consolidado ────────────────────────────────────────────
    print("\n[RESUMO] Gráfico consolidado...")
    summary = plot_consolidated(train_df, test_df, val_df)
    print("\n" + summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("  PIPELINE CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/BASE_DE_DADOS_PEDE__-_DATATHON.xlsx"
    main(path)
