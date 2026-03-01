"""
Rotas da API: /predict, /health, /metrics, /models.
"""
import time
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.metrics import (
    predictions_total,
    prediction_latency_seconds,
    prediction_probability,
    prediction_errors_total,
)

router = APIRouter()


class ModelName(str, Enum):
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    gradient_boosting = "gradient_boosting"
    neural_network = "neural_network"


# =====================================================================
# Schemas
# =====================================================================


class StudentInput(BaseModel):
    """Dados de entrada de um aluno para predição de defasagem escolar."""

    idade: float = Field(
        ...,
        description="Idade do aluno (valor bruto ou z-score, dependendo do pré-processamento).",
        examples=[12],
    )
    genero_masculino: int = Field(
        0, ge=0, le=1,
        description="Gênero do aluno. 1 = masculino, 0 = feminino.",
        examples=[1],
    )
    instituicao_publica: int = Field(
        1, ge=0, le=1,
        description="Tipo de instituição de ensino. 1 = pública, 0 = privada.",
        examples=[1],
    )
    num_av: float = Field(
        3.0,
        description="Número de avaliações realizadas pelo aluno.",
        examples=[3],
    )
    iaa: float = Field(
        ...,
        description="IAA — Indicador de Auto Avaliação.",
        examples=[7.5],
    )
    ieg: float = Field(
        ...,
        description="IEG — Indicador de Engajamento.",
        examples=[6.0],
    )
    ips: float = Field(
        ...,
        description="IPS — Indicador Psicossocial.",
        examples=[5.8],
    )
    ida: float = Field(
        ...,
        description="IDA — Indicador de Aprendizagem.",
        examples=[4.5],
    )
    ipv: float = Field(
        ...,
        description="IPV — Indicador de Ponto de Virada.",
        examples=[6.2],
    )
    ian: float = Field(
        ...,
        description="IAN — Indicador de Adequação de Nível.",
        examples=[5.0],
    )
    matematica: float = Field(
        ...,
        description="Nota de Matemática do aluno.",
        examples=[5.0],
    )
    fase: int = Field(
        1, ge=0, le=7,
        description="Fase educacional do aluno (0 = ALFA, 1-7 = FASE 1 a 7).",
        examples=[3],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "idade": 12, "genero_masculino": 1, "instituicao_publica": 1,
                    "num_av": 3, "iaa": 7.5, "ieg": 6.0, "ips": 5.8,
                    "ida": 4.5, "ipv": 6.2, "ian": 5.0, "matematica": 5.0, "fase": 3,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Resultado da predição de defasagem escolar."""

    prediction: int = Field(
        ...,
        description="Classe predita. 0 = Não Defasado, 1 = Defasado.",
        examples=[0],
    )
    label: str = Field(
        ...,
        description="Rótulo legível da predição.",
        examples=["Não Defasado"],
    )
    probability: float = Field(
        ...,
        description="Probabilidade (0-1) do aluno estar defasado.",
        examples=[0.2345],
    )
    risk_level: str = Field(
        ...,
        description="Nível de risco: Baixo (< 0.3), Moderado (0.3-0.7) ou Alto (≥ 0.7).",
        examples=["Baixo"],
    )
    model_name: str = Field(
        ...,
        description="Nome do modelo utilizado na predição.",
        examples=["logistic_regression"],
    )
    timestamp: str = Field(
        ...,
        description="Data/hora da predição em formato ISO 8601.",
        examples=["2026-02-28T17:30:00.000000"],
    )


class HealthResponse(BaseModel):
    """Status de saúde da API."""

    status: str = Field(
        ...,
        description="Estado geral da API: 'healthy' ou 'degraded'.",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Indica se ao menos um modelo ML está carregado em memória.",
        examples=[True],
    )
    models_loaded: int = Field(
        ...,
        description="Quantidade de modelos carregados em memória.",
        examples=[4],
    )
    predictions: int = Field(
        ...,
        description="Total de predições realizadas desde o início da aplicação.",
        examples=[42],
    )


class ErrorResponse(BaseModel):
    """Resposta de erro padrão da API."""

    detail: str = Field(
        ...,
        description="Mensagem descritiva do erro.",
        examples=["Modelo não carregado."],
    )


class RootResponse(BaseModel):
    """Informações básicas da API."""

    api: str = Field(..., examples=["Passos Mágicos"])
    version: str = Field(..., examples=["1.0.0"])
    docs: str = Field(..., examples=["/docs"])


# =====================================================================
# Endpoints
# =====================================================================


@router.get(
    "/",
    response_model=RootResponse,
    tags=["Monitoramento"],
    summary="Informações da API",
    description="Retorna nome, versão e link para a documentação Swagger.",
)
def root():
    return {"api": "Passos Mágicos", "version": "1.0.0", "docs": "/docs"}


@router.get(
    "/metrics",
    tags=["Monitoramento"],
    summary="Métricas Prometheus",
    description=(
        "Exporta métricas no formato Prometheus para scraping. "
        "Inclui contadores de predições, latência, distribuição de probabilidades, "
        "erros e status do modelo."
    ),
    responses={
        200: {
            "content": {"text/plain": {}},
            "description": "Métricas no formato Prometheus text exposition.",
        }
    },
)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoramento"],
    summary="Health check",
    description=(
        "Verifica o estado da aplicação. Retorna `healthy` se o modelo "
        "está carregado, `degraded` caso contrário. Utilizado pelo "
        "Docker HEALTHCHECK."
    ),
)
def health(request: Request):
    state = request.app.state.app_state
    loaded = len(state["models"])
    return {
        "status": "healthy" if loaded > 0 else "degraded",
        "model_loaded": loaded > 0,
        "models_loaded": loaded,
        "predictions": state["prediction_count"],
    }


@router.get(
    "/models",
    tags=["Monitoramento"],
    summary="Modelos disponíveis",
    description="Lista os modelos de ML carregados e indica qual é o modelo padrão.",
)
def list_models(request: Request):
    state = request.app.state.app_state
    return {
        "default": state["default_model"],
        "available": list(state["models"].keys()),
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predição"],
    summary="Predição de defasagem escolar",
    description=(
        "Recebe os dados de um aluno e retorna a predição de defasagem "
        "escolar. Use o query parameter `model` para escolher o modelo "
        "(padrão: `logistic_regression`).\n\n"
        "**Modelos disponíveis:** `logistic_regression`, `random_forest`, "
        "`gradient_boosting`, `neural_network`\n\n"
        "**Fluxo interno:**\n"
        "1. Monta feature vector com one-hot encoding da fase (fase_0 a fase_7)\n"
        "2. Alinha colunas com as features esperadas pelo modelo\n"
        "3. Executa predição e calcula probabilidade\n"
        "4. Classifica nível de risco com base na probabilidade\n"
        "5. Registra métricas Prometheus (latência, probabilidade, contagem)"
    ),
    responses={
        200: {
            "description": "Predição realizada com sucesso.",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 0,
                        "label": "Não Defasado",
                        "probability": 0.2345,
                        "risk_level": "Baixo",
                        "timestamp": "2026-02-28T17:30:00.000000",
                    }
                }
            },
        },
        503: {
            "model": ErrorResponse,
            "description": "Modelo ML não está carregado. A API está em modo degradado.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Erro interno durante a execução da predição.",
        },
    },
)
def predict(
    student: StudentInput,
    request: Request,
    model: Optional[ModelName] = Query(
        None,
        description="Modelo a ser utilizado na predição. Se não informado, usa o modelo padrão (logistic_regression).",
    ),
):
    start = time.perf_counter()
    state = request.app.state.app_state
    model_name = model.value if model else state["default_model"]

    if model_name not in state["models"]:
        prediction_errors_total.inc()
        raise HTTPException(404, f"Modelo '{model_name}' não disponível.")

    try:
        # Montar feature vector com dummies de fase
        features = {
            "idade": student.idade, "genero.masculino": student.genero_masculino,
            "instituicao.publica": student.instituicao_publica, "nº.av": student.num_av,
            "iaa": student.iaa, "ieg": student.ieg, "ips": student.ips,
            "ida": student.ida, "ipv": student.ipv, "ian": student.ian,
            "matematica": student.matematica,
        }
        for i in range(8):
            features[f"fase_{i}"] = 1 if student.fase == i else 0

        import pandas as pd
        X = pd.DataFrame([features])

        # Alinhar colunas com o modelo
        selected_model = state["models"][model_name]
        if hasattr(selected_model, "feature_names_in_"):
            for c in selected_model.feature_names_in_:
                if c not in X.columns:
                    X[c] = 0
            X = X[selected_model.feature_names_in_]

        pred = int(selected_model.predict(X)[0])
        prob = float(selected_model.predict_proba(X)[0][1])
        risk = "Baixo" if prob < 0.3 else ("Moderado" if prob < 0.7 else "Alto")
        label = "Defasado" if pred == 1 else "Não Defasado"

        state["prediction_count"] += 1

        # Métricas Prometheus
        prediction_latency_seconds.observe(time.perf_counter() - start)
        prediction_probability.observe(prob)
        predictions_total.labels(risk_level=risk, prediction=label).inc()

        return PredictionResponse(
            prediction=pred,
            label=label,
            probability=round(prob, 4),
            risk_level=risk,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception:
        prediction_errors_total.inc()
        raise HTTPException(500, "Erro interno na predição.")
