"""API de predicción: sirve los modelos entrenados por scripts/train_dataset.py.

Recibe un vector de features ya procesado (mismas columnas que artifact["features"]
para el nivel/target pedidos) y devuelve la predicción puntual del modelo. No
reconstruye features desde datos crudos -- ver docs/informe.md §9 sobre por qué eso
queda fuera de alcance por ahora (requiere recalcular en tiempo real ~140 columnas de
lags/rolling/target-encoding, hoy solo se hace en backtest sobre el parquet histórico).

Cada level_id tiene dos granularidades posibles (`grain`: "daily"/"weekly", ver
config.Level.grains) y por defecto se entrenan ambas -- si existen artifacts de las
dos para el mismo level_id+target, `grain` es obligatorio en /predict y /levels
filtra por él (sin especificarlo, /predict responde 400 en vez de elegir una al
azar, ver api/registry.py).

Levantar con: make serve-api  (o `uv run uvicorn api.main:app --reload`)
"""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

import config
from api.registry import GRAINS, artifact_grain, get_model, load_registry, resolve_version
from api.schemas import LevelInfo, PredictRequest, PredictResponse
from src.logging_setup import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("api")
    logger.info("API iniciada")
    yield


app = FastAPI(title="Forecast API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/levels", response_model=list[LevelInfo])
def list_levels(target: str = "sales", grain: str | None = None) -> list[LevelInfo]:
    if grain is not None and grain not in GRAINS:
        raise HTTPException(status_code=400, detail=f"grain inválido: {grain!r} (debe ser uno de {GRAINS})")

    out = []
    for level_id in config.ACTIVE_LEVEL_IDS:
        # Se prueba cada grain por separado (en vez de pedirle a get_model el default)
        # para listar una fila por cada uno que tenga artifact, sin nunca dejar que
        # get_model/resolve_version tengan que desambiguar entre daily y weekly.
        for candidate_grain in ([grain] if grain else GRAINS):
            try:
                artifact = get_model(level_id, target, grain=candidate_grain)
            except (FileNotFoundError, KeyError):
                continue
            out.append(LevelInfo(
                level_id=level_id,
                level=artifact["level"],
                grain=candidate_grain,
                target=target,
                version=resolve_version(level_id, target, grain=candidate_grain),
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
def predict(level_id: int, body: PredictRequest, target: str = "sales", grain: str | None = None) -> PredictResponse:
    try:
        artifact = get_model(level_id, target, body.version, grain)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        # grain inválido, o ambiguo (existen ambos grains y no se especificó cuál) --
        # ver api/registry.py::_registry_key.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X = _build_feature_row(body.features, artifact)
    prediction = float(artifact["model"].predict(X)[0])
    resolved_grain = artifact_grain(artifact)
    version = resolve_version(level_id, target, body.version, resolved_grain)
    return PredictResponse(level_id=level_id, grain=resolved_grain, target=target, version=version, prediction=prediction)
