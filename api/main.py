"""API de predicción: sirve los modelos entrenados por scripts/train_dataset.py.

Recibe un vector de features ya procesado (mismas columnas que artifact["features"]
para el nivel/target pedidos) y devuelve la predicción puntual del modelo. No
reconstruye features desde datos crudos -- ver docs/informe.md §9 sobre por qué eso
queda fuera de alcance por ahora (requiere recalcular en tiempo real ~140 columnas de
lags/rolling/target-encoding, hoy solo se hace en backtest sobre el parquet histórico).

Levantar con: make serve-api  (o `uv run uvicorn api.main:app --reload`)
"""
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from loguru import logger

import config
from api.registry import get_model, load_registry, resolve_version
from api.schemas import LevelInfo, PredictRequest, PredictResponse
from src.logging_setup import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("api")
    logger.info("API iniciada")
    yield


app = FastAPI(title="Forecast TFM API", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("{} {} -> {} ({:.1f} ms)", request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/levels", response_model=list[LevelInfo])
def list_levels(target: str = "sales") -> list[LevelInfo]:
    out = []
    for level_id in config.ACTIVE_LEVEL_IDS:
        try:
            artifact = get_model(level_id, target)
        except (FileNotFoundError, KeyError):
            continue
        out.append(LevelInfo(
            level_id=level_id,
            level=artifact["level"],
            target=target,
            version=resolve_version(level_id, target),
            wape_test=artifact.get("wape_test"),
            wrmsse_test=artifact.get("wrmsse_test"),
        ))
    return out


def _build_feature_row(features: dict, artifact: dict) -> pd.DataFrame:
    expected = set(artifact["features"])
    got = set(features)
    if got != expected:
        raise HTTPException(status_code=400, detail={
            "error": "columnas de features no coinciden con las esperadas por el modelo",
            "faltantes": sorted(expected - got),
            "sobrantes": sorted(got - expected),
        })

    X = pd.DataFrame([features])[artifact["features"]]
    trained_categories = artifact.get("categorical_categories", {})
    for col in artifact["categorical_features"]:
        # astype("category") sobre 1 fila con valor nulo (p.ej. event_name_2, casi
        # siempre nulo) deja categories=[]: XGBoost lo rechaza ("must have at least one
        # category"). Un placeholder inventado tampoco sirve -- XGBoost valida cada
        # categoría declarada contra las vistas en training, la use o no la fila. Por
        # eso hace falta el dominio real (guardado en el artifact desde que existe
        # categorical_categories, ver scripts/train_dataset/export.py::export_artifact);
        # el valor de la fila sigue siendo NaN (missing) pase lo que pase.
        value = X[col].iloc[0]
        if col in trained_categories:
            categories = trained_categories[col]
        else:
            # Artifact viejo, sin categorical_categories: mismo comportamiento que antes
            # (astype("category") desde una sola fila) -- puede fallar en XGBoost si esta
            # fila trae un valor faltante en `col`, ver comentario arriba.
            categories = [value] if pd.notna(value) else []
        X[col] = pd.Categorical(X[col], categories=categories)
    numerical_cols = [c for c in artifact["features"] if c not in artifact["categorical_features"]]
    try:
        X[numerical_cols] = X[numerical_cols].apply(pd.to_numeric)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"columna numérica inválida: {exc}") from exc
    return X


@app.post("/predict/{level_id}", response_model=PredictResponse)
def predict(level_id: int, body: PredictRequest, target: str = "sales") -> PredictResponse:
    try:
        artifact = get_model(level_id, target, body.version)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    X = _build_feature_row(body.features, artifact)
    prediction = float(artifact["model"].predict(X)[0])
    version = resolve_version(level_id, target, body.version)
    return PredictResponse(level_id=level_id, target=target, version=version, prediction=prediction)
