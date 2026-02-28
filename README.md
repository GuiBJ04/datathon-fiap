# Passos Magicos — Predicao de Defasagem Escolar

Modelo preditivo de Machine Learning para identificar risco de piora na defasagem escolar dos alunos da Associacao Passos Magicos, utilizando dados do programa PEDE (2022-2024).

## Indice

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura](#arquitetura)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalacao](#instalacao)
- [Como Executar](#como-executar)
  - [Pipeline de Treinamento](#pipeline-de-treinamento)
  - [Analise Exploratoria](#analise-exploratoria)
  - [API de Predicao](#api-de-predicao)
- [API — Documentacao](#api--documentacao)
  - [Endpoints](#endpoints)
  - [Exemplo de Uso](#exemplo-de-uso)
  - [Swagger UI](#swagger-ui)
- [Docker](#docker)
- [Monitoramento](#monitoramento)
- [Pipeline de ML](#pipeline-de-ml)
  - [Pre-processamento](#pre-processamento)
  - [Modelagem](#modelagem)
  - [Features](#features)
- [Testes](#testes)
- [Estrutura de Saida](#estrutura-de-saida)

---

## Sobre o Projeto

A Associacao Passos Magicos atua na transformacao da vida de criancas e jovens em situacao de vulnerabilidade social. Este projeto utiliza dados educacionais do programa PEDE para prever quais alunos tem risco de aumento na defasagem escolar, permitindo intervencao preventiva.

**Objetivo:** dado o perfil de um aluno em um ano, prever se sua defasagem escolar ira piorar no ano seguinte.

| Item | Detalhe |
|------|---------|
| Target | `defasagem` (0 = Nao Defasado, 1 = Defasado) |
| Dados | PEDE 2022, 2023, 2024 (Excel com 3 abas) |
| Treino | Alunos 2022 avaliados em 2023 |
| Validacao | Alunos novos de 2023 avaliados em 2024 |
| Modelos | Random Forest, Logistic Regression, Gradient Boosting, Neural Network |

---

## Arquitetura

```
                    +-------------------+
                    |  BASE_DE_DADOS    |
                    |  PEDE.xlsx        |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  main.py          |
                    |  Pipeline ML      |
                    |  (11 etapas)      |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+       +-----------v-----------+
    |  output/          |       |  app/model/           |
    |  CSVs + PNGs      |       |  *.pkl (modelos)      |
    +-------------------+       +-----------+-----------+
                                            |
                                +-----------v-----------+
                                |  FastAPI (app/)       |
                                |  POST /predict        |
                                +-----------+-----------+
                                            |
                                +-----------v-----------+
                                |  Prometheus + Grafana  |
                                |  (monitoring/)        |
                                +-----------------------+
```

---

## Estrutura do Projeto

```
passos-magicos/
├── main.py                     # Pipeline principal (11 etapas)
├── analysis.py                 # Analise exploratoria + feature importance
├── requirements.txt            # Dependencias Python
├── Dockerfile                  # Imagem Docker da API
├── docker-compose.yml          # Stack completa (API + Prometheus + Grafana)
│
├── src/                        # Modulos do pipeline de ML
│   ├── preprocessing.py        # Carga, limpeza e preparacao dos dados
│   ├── train.py                # Treinamento com GridSearchCV + 5-fold CV
│   ├── evaluate.py             # Calculo de metricas
│   ├── visualize.py            # Geracao de graficos
│   └── utils.py                # Funcoes auxiliares e constantes
│
├── app/                        # API FastAPI
│   ├── main.py                 # Inicializacao da app e middleware
│   ├── routes.py               # Endpoints (/predict, /health, /metrics)
│   ├── metrics.py              # Metricas Prometheus
│   └── model/                  # Modelos treinados (.pkl)
│
├── monitoring/                 # Observabilidade
│   ├── prometheus.yml          # Configuracao do Prometheus
│   ├── drift_monitor.py        # Deteccao de data drift (teste KS)
│   └── grafana/
│       ├── dashboards/         # Dashboard pre-configurado
│       └── provisioning/       # Datasources e provisioning
│
├── tests/                      # Suite de testes
│   ├── conftest.py             # Fixtures compartilhadas
│   ├── test_api.py             # Testes da API
│   ├── test_preprocessing.py   # Testes de pre-processamento
│   ├── test_train.py           # Testes de treinamento
│   ├── test_evaluate.py        # Testes de avaliacao
│   ├── test_visualize.py       # Testes de visualizacao
│   ├── test_utils.py           # Testes de utilidades
│   └── test_pipeline.py        # Testes de integracao
│
├── data/                       # Dados de entrada (Excel)
├── output/                     # CSVs, graficos e resultados
└── logs/                       # Logs da aplicacao
```

---

## Requisitos

- Python 3.12+
- Docker e Docker Compose (para deploy containerizado)

**Dependencias principais:**

| Pacote | Versao | Uso |
|--------|--------|-----|
| pandas | 2.2.2 | Manipulacao de dados |
| numpy | >= 2.0.0 | Operacoes numericas |
| scikit-learn | >= 1.8.0 | Modelos de ML |
| matplotlib | 3.9.2 | Graficos |
| seaborn | 0.13.2 | Graficos estatisticos |
| fastapi | 0.115.0 | API REST |
| uvicorn | 0.30.6 | Servidor ASGI |
| prometheus-client | 0.21.0 | Metricas |
| openpyxl | 3.1.5 | Leitura de Excel |
| scipy | >= 1.14.1 | Testes estatisticos |

---

## Instalacao

```bash
# Clonar o repositorio
git clone <url-do-repositorio>
cd passos-magicos

# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Colocar o arquivo Excel na pasta data/
cp BASE_DE_DADOS_PEDE.xlsx data/
```

---

## Como Executar

### Pipeline de Treinamento

Executa todo o pipeline: pre-processamento, treinamento dos 4 modelos com GridSearchCV, avaliacao e geracao de graficos.

```bash
python main.py

# Ou com caminho customizado para o Excel
python main.py --data caminho/para/arquivo.xlsx
```

**Saida:** modelos `.pkl` em `app/model/`, CSVs e graficos em `output/`.

### Analise Exploratoria

Gera graficos descritivos e de feature importance a partir dos dados e modelos ja existentes, sem re-treinar.

```bash
python analysis.py
```

**Graficos gerados em `output/`:**

| Arquivo | Conteudo |
|---------|----------|
| `descritivo_missing.png` | Dados faltantes por coluna (dados brutos, 3 anos) |
| `descritivo_target.png` | Distribuicao de classes (defasagem 0 vs 1) |
| `descritivo_features.png` | Histogramas das 9 features numericas por classe |
| `descritivo_correlacao.png` | Heatmap de correlacao |
| `feature_importance.png` | Importancia de features nos 4 modelos |

### API de Predicao

```bash
# Execucao local
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Ou via Docker
docker compose up -d
```

A API estara disponivel em `http://localhost:8000`.

---

## API — Documentacao

### Endpoints

| Metodo | Rota | Descricao |
|--------|------|-----------|
| `GET` | `/` | Informacoes da API (nome, versao, link para docs) |
| `GET` | `/health` | Health check — status do modelo e contagem de predicoes |
| `GET` | `/metrics` | Metricas Prometheus (text exposition) |
| `POST` | `/predict` | Predicao de defasagem escolar |

### Exemplo de Uso

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 12,
    "genero_masculino": 1,
    "instituicao_publica": 1,
    "num_av": 3,
    "iaa": 7.5,
    "ieg": 6.0,
    "ips": 5.8,
    "ida": 4.5,
    "ipv": 6.2,
    "ian": 5.0,
    "matematica": 5.0,
    "fase": 3
  }'
```

**Response:**

```json
{
  "prediction": 0,
  "label": "Nao Defasado",
  "probability": 0.2345,
  "risk_level": "Baixo",
  "timestamp": "2026-02-28T17:30:00.000000"
}
```

**Campos de entrada:**

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `idade` | float | Idade do aluno |
| `genero_masculino` | int (0-1) | 1 = masculino, 0 = feminino |
| `instituicao_publica` | int (0-1) | 1 = publica, 0 = privada |
| `num_av` | float | Numero de avaliacoes |
| `iaa` | float | Indicador de Auto Avaliacao |
| `ieg` | float | Indicador de Engajamento |
| `ips` | float | Indicador Psicossocial |
| `ida` | float | Indicador de Aprendizagem |
| `ipv` | float | Indicador de Ponto de Virada |
| `ian` | float | Indicador de Adequacao de Nivel |
| `matematica` | float | Nota de Matematica |
| `fase` | int (0-7) | Fase educacional (0 = ALFA, 1-7 = FASE 1 a 7) |

**Niveis de risco:**

| Probabilidade | Nivel |
|---------------|-------|
| < 0.3 | Baixo |
| 0.3 - 0.7 | Moderado |
| >= 0.7 | Alto |

**Codigos de erro:**

| Codigo | Descricao |
|--------|-----------|
| 422 | Dados de entrada invalidos |
| 503 | Modelo nao carregado (API em modo degradado) |
| 500 | Erro interno na predicao |

### Swagger UI

A documentacao interativa da API esta disponivel em:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Docker

### Build e execucao da API isolada

```bash
docker build -t passos-magicos .
docker run -p 8000:8000 passos-magicos
```

### Stack completa (API + Prometheus + Grafana)

```bash
docker compose up -d
```

| Servico | Porta | URL |
|---------|-------|-----|
| API FastAPI | 8000 | http://localhost:8000 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |

**Credenciais Grafana:** `admin` / `admin`

```bash
# Parar todos os servicos
docker compose down

# Rebuild apos alteracoes no codigo
docker compose up -d --build
```

A imagem Docker inclui health check automatico (`GET /health` a cada 30s).

---

## Monitoramento

A stack de observabilidade inclui:

**Metricas Prometheus** (expostas em `/metrics`):

| Metrica | Tipo | Descricao |
|---------|------|-----------|
| `predictions_total` | Counter | Total de predicoes por nivel de risco e classe |
| `prediction_latency_seconds` | Histogram | Latencia do endpoint /predict |
| `prediction_probability` | Histogram | Distribuicao de probabilidades preditas |
| `prediction_errors_total` | Counter | Total de erros de predicao |
| `http_requests_total` | Counter | Requests HTTP por metodo, endpoint e status |
| `model_loaded` | Gauge | 1 se modelo carregado, 0 caso contrario |
| `data_drift_features_total` | Gauge | Features com drift detectado |

**Dashboard Grafana** pre-configurado com:
- Requests/s por endpoint
- Latencia do /predict (p50, p95, p99)
- Distribuicao de predicoes por classe e nivel de risco
- Status do modelo e contagem de erros

**Deteccao de Data Drift** (`monitoring/drift_monitor.py`):
- Teste Kolmogorov-Smirnov por feature
- Gera dashboard HTML com status de drift

---

## Pipeline de ML

### Pre-processamento

1. Normalizacao de nomes de colunas (lowercase, sem acentos)
2. Alinhamento de colunas entre datasets
3. Criacao do target binario (`defasagem_final - defasagem_inicial < 0`)
4. Remocao de features pouco informativas (avaliadores, turma, etc.)
5. Tratamento da coluna fase (ALFA -> 0, FASE N -> N)
6. Hot encoding (instituicao publica, genero masculino)
7. Remocao de colunas com > 50% de valores faltantes
8. Dummy encoding de fase (fase_0 a fase_7)
9. Reordenacao de colunas
10. Padronizacao z-score das features numericas

### Modelagem

11 etapas executadas por `main.py`:

| Etapa | Descricao |
|-------|-----------|
| 1 | Split 70/30 estratificado |
| 2-5 | Grid Search com 5-fold CV para 4 modelos |
| 6 | Salvamento dos modelos otimos e hiperparametros |
| 7 | Avaliacao e graficos nos dados de treino |
| 8-9 | Avaliacao e graficos nos dados de teste |
| 10-11 | Avaliacao e graficos nos dados de validacao |

**Modelos e hiperparametros explorados:**

| Modelo | Hiperparametros |
|--------|----------------|
| Random Forest | n_estimators: [100, 200, 300], max_depth: [3, 5, 8, None], min_samples_leaf: [1, 3, 5] |
| Logistic Regression | C: [0.001-10], penalty: [l1, l2], solver: saga |
| Gradient Boosting | n_estimators: [100-300], max_depth: [3-5], learning_rate: [0.01-0.1] |
| Neural Network (MLP) | hidden_layers: [(64,32), (128,64), (64,32,16)], alpha: [0.0001-0.01] |

O modelo **Gradient Boosting** e utilizado na API de producao.

### Features

19 features apos pre-processamento:

| # | Feature | Tipo | Descricao |
|---|---------|------|-----------|
| 1 | `idade` | float (z-score) | Idade do aluno |
| 2 | `genero.masculino` | int (0/1) | Genero |
| 3 | `instituicao.publica` | int (0/1) | Tipo de escola |
| 4 | `nº.av` | float (z-score) | Numero de avaliacoes |
| 5 | `iaa` | float (z-score) | Indicador de Auto Avaliacao |
| 6 | `ieg` | float (z-score) | Indicador de Engajamento |
| 7 | `ips` | float (z-score) | Indicador Psicossocial |
| 8 | `ida` | float (z-score) | Indicador de Aprendizagem |
| 9 | `ipv` | float (z-score) | Indicador de Ponto de Virada |
| 10 | `ian` | float (z-score) | Indicador de Adequacao de Nivel |
| 11 | `matematica` | float (z-score) | Nota de Matematica |
| 12-19 | `fase_0` a `fase_7` | int (0/1) | Fase educacional (one-hot) |

---

## Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ -v --cov=src --cov=app
```

A suite de testes cobre:

| Modulo | Arquivo de Teste | Cobertura |
|--------|-----------------|-----------|
| Pre-processamento | `test_preprocessing.py` | Carga, normalizacao, encoding, scaling |
| Treinamento | `test_train.py` | Split, GridSearchCV, salvamento |
| Avaliacao | `test_evaluate.py` | Calculo de metricas |
| Visualizacao | `test_visualize.py` | Geracao de graficos |
| Utilidades | `test_utils.py` | Logger, load/save modelo |
| API | `test_api.py` | Endpoints, respostas, estado |
| Integracao | `test_pipeline.py` | Pipeline completo |

---

## Estrutura de Saida

Apos execucao completa do pipeline (`main.py`) e analise (`analysis.py`):

```
output/
├── training_set.csv                # Dados de treino preprocessados
├── validation_set.csv              # Dados de validacao preprocessados
├── resultados_completos.csv        # Metricas de todos os modelos
│
├── etapa7_comparacao_treino.png    # Comparacao de modelos (treino)
├── etapa7_roc_treino.png           # Curvas ROC (treino)
├── etapa7_confusion_treino.png     # Matrizes de confusao (treino)
├── etapa9_comparacao_teste.png     # Comparacao de modelos (teste)
├── etapa9_roc_teste.png            # Curvas ROC (teste)
├── etapa9_confusion_teste.png      # Matrizes de confusao (teste)
├── etapa11_comparacao_validacao.png # Comparacao de modelos (validacao)
├── etapa11_roc_validacao.png       # Curvas ROC (validacao)
├── etapa11_confusion_validacao.png # Matrizes de confusao (validacao)
├── resumo_consolidado.png          # Resumo treino vs teste vs validacao
│
├── descritivo_missing.png          # Dados faltantes (dados brutos)
├── descritivo_target.png           # Distribuicao do target
├── descritivo_features.png         # Histogramas das features
├── descritivo_correlacao.png       # Heatmap de correlacao
└── feature_importance.png          # Importancia de features (4 modelos)
```
