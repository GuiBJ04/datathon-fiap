"""
API para predição de defasagem escolar — FastAPI.
"""
import os, sys
from contextlib import asynccontextmanager
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
from app.metrics import model_loaded, model_info, http_requests_total
from src.utils import load_model, setup_logger

logger = setup_logger("api")

AVAILABLE_MODELS = ["logistic_regression", "random_forest", "gradient_boosting", "neural_network"]
DEFAULT_MODEL = "logistic_regression"

app_state = {
    "models": {},
    "default_model": DEFAULT_MODEL,
    "start_time": None,
    "prediction_count": 0,
}

API_DESCRIPTION = """
## Sobre

API de Machine Learning do projeto **Passos Mágicos** para predição de
defasagem escolar. Disponibiliza múltiplos modelos treinados com dados
do programa PEDE (2022-2024) para identificar alunos com risco de
defasagem.

## Funcionalidades

* **Predição** — Envia dados de um aluno e recebe a probabilidade de
  defasagem, classificação (Defasado / Não Defasado) e nível de risco.
  É possível escolher o modelo via query parameter `model`.
* **Modelos** — Endpoint `/models` lista os modelos disponíveis e o
  modelo padrão.
* **Monitoramento** — Métricas Prometheus para latência, volume de
  predições, distribuição de probabilidades e status do modelo.
* **Health Check** — Endpoint para verificação de saúde da aplicação,
  integrado ao Docker `HEALTHCHECK`.

## Modelos disponíveis

| Modelo | Padrão |
|--------|--------|
| Logistic Regression | ✅ |
| Random Forest | |
| Gradient Boosting | |
| Neural Network | |

Features: 19 (9 indicadores + gênero + instituição + 8 fases one-hot)
Target: `defasagem` (0 = Não Defasado, 1 = Defasado)
Treino: Dados PEDE 2022 → 2023

## Níveis de Risco

| Probabilidade | Nível |
|---------------|-------|
| < 0.3 | Baixo |
| 0.3 – 0.7 | Moderado |
| ≥ 0.7 | Alto |
"""

TAGS_METADATA = [
    {
        "name": "Predição",
        "description": "Endpoint principal de predição de defasagem escolar.",
    },
    {
        "name": "Monitoramento",
        "description": "Health check e métricas Prometheus para observabilidade.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in AVAILABLE_MODELS:
        try:
            app_state["models"][name] = load_model(f"{name}.pkl")
            logger.info(f"Modelo '{name}' carregado!")
        except Exception as e:
            logger.warning(f"Falha ao carregar modelo '{name}': {e}")

    loaded = len(app_state["models"])
    model_loaded.set(1 if loaded > 0 else 0)
    model_info.info({"name": DEFAULT_MODEL, "version": "1.0.0", "total_loaded": str(loaded)})
    app_state["start_time"] = datetime.now()
    yield


app = FastAPI(
    title="Passos Mágicos — Predição de Defasagem Escolar",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "Equipe Passos Mágicos",
        "url": "https://github.com/passos-magicos",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.state.app_state = app_state
app.include_router(router)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    response: Response = await call_next(request)
    if request.url.path != "/metrics":
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
    return response
