"""
Métricas Prometheus centralizadas para a API Passos Mágicos.
"""
from prometheus_client import Counter, Histogram, Gauge, Info

# Predições
predictions_total = Counter(
    "predictions_total",
    "Total de predições realizadas",
    ["risk_level", "prediction"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Latência do endpoint /predict em segundos",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

prediction_probability = Histogram(
    "prediction_probability",
    "Distribuição de probabilidades preditas",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total de erros no endpoint de predição",
)

# HTTP
http_requests_total = Counter(
    "http_requests_total",
    "Total de requests HTTP",
    ["method", "endpoint", "status_code"],
)

# Modelo
model_loaded = Gauge(
    "model_loaded",
    "1 se o modelo está carregado, 0 caso contrário",
)

model_info = Info(
    "model",
    "Metadados do modelo carregado",
)

# Drift
data_drift_features_total = Gauge(
    "data_drift_features_total",
    "Quantidade de features com drift detectado",
)
