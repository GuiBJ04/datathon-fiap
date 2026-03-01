#!/bin/bash
set -e

if [ "$MODE" = "pipeline" ]; then
    echo "=== Executando Pipeline de ML ==="
    python main.py --data "${DATA_PATH:-data/BASE_DE_DADOS_PEDE.xlsx}"

    # Copiar modelos para o diretorio da API
    echo "Copiando modelos para app/model/..."
    cp models/*.pkl app/model/
    cp models/*.json app/model/ 2>/dev/null || true

    echo "=== Executando Analise Exploratoria ==="
    python analysis.py

    echo "=== Pipeline concluido! ==="
else
    # Garantir que modelos pre-treinados estejam disponiveis
    echo "Copiando modelos pre-treinados para app/model/..."
    cp models/*.pkl app/model/ 2>/dev/null || true
    cp models/*.json app/model/ 2>/dev/null || true

    exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
fi
