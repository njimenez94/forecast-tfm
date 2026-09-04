"""API de predicción: sirve los modelos entrenados por scripts/train_dataset.py.

Recibe un vector de features ya procesado (mismas columnas que artifact["features"]
para el nivel/target pedidos) y devuelve la predicción puntual del modelo. No
reconstruye features desde datos crudos -- ver docs/informe.md §9 sobre por qué eso
queda fuera de alcance por ahora (requiere recalcular en tiempo real ~140 columnas de
lags/rolling/target-encoding, hoy solo se hace en backtest sobre el parquet histórico).

Levantar con: make serve-api  (o `uv run uvicorn api.main:app --reload`)
"""
import pandas as pd
from fastapi import FastAPI, HTTPException

import config
from api.registry import get_model, load_registry, resolve_version
from api.schemas import LevelInfo, PredictRequest, PredictResponse

app = FastAPI(title="Forecast TFM API")


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


@app.post("/predict/{level_id}", response_model=PredictResponse)
def predict(level_id: int, body: PredictRequest, target: str = "sales") -> PredictResponse:
    try:
        artifact = get_model(level_id, target, body.version)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    expected = set(artifact["features"])
    got = set(body.features)
    if got != expected:
        raise HTTPException(status_code=400, detail={
            "error": "columnas de features no coinciden con las esperadas por el modelo",
            "faltantes": sorted(expected - got),
            "sobrantes": sorted(got - expected),
        })

    X = pd.DataFrame([body.features])[artifact["features"]]
    for col in artifact["categorical_features"]:
        X[col] = X[col].astype("category")
    numerical_cols = [c for c in artifact["features"] if c not in artifact["categorical_features"]]
    try:
        X[numerical_cols] = X[numerical_cols].apply(pd.to_numeric)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"columna numérica inválida: {exc}") from exc

    prediction = float(artifact["model"].predict(X)[0])
    version = resolve_version(level_id, target, body.version)
    return PredictResponse(level_id=level_id, target=target, version=version, prediction=prediction)
