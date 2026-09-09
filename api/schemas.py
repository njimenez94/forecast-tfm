"""Schemas Pydantic de la API de predicción."""
from pydantic import BaseModel


class PredictRequest(BaseModel):
    # Vector de features ya procesado, con las mismas columnas que
    # artifact["features"] para el nivel/target pedidos (ver notebooks/03_predictions.ipynb
    # y src.data.temporal_split.build_feature_matrices para cómo se arma ese vector en training/backtest).
    features: dict[str, float | int | str | None]
    version: str | None = None


class PredictResponse(BaseModel):
    level_id: int
    target: str
    version: str
    prediction: float


class LevelInfo(BaseModel):
    level_id: int
    level: str
    target: str
    version: str | None
    wape_test: float | None
    wrmsse_test: float | None
