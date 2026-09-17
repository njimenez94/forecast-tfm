"""Schemas Pydantic de la API de predicción."""
import datetime

from pydantic import BaseModel


class PredictRequest(BaseModel):
    # Vector de features ya procesado, con las mismas columnas que
    # artifact["features"] para el nivel/target pedidos (ver notebooks/03_predictions.ipynb
    # y src.data.temporal_split.build_feature_matrices para cómo se arma ese vector en training/backtest).
    features: dict[str, float | int | str | None]
    version: str | None = None


class PredictResponse(BaseModel):
    level_id: int
    grain: str  # "daily"|"weekly" -- el que realmente se sirvió (artifact["level_label"]),
    # no necesariamente el que se pidió: si `grain` no viene en la request y solo hay
    # un grain entrenado para ese level_id+target, se resuelve solo; si hay dos, la
    # request es rechazada (400) y hay que pedir uno explícito, ver api/registry.py.
    target: str
    version: str
    prediction: float


class ForecastRequest(BaseModel):
    # A diferencia de PredictRequest (vector de features ya calculado), esto es
    # lo que un cliente real tendría a mano: qué serie y qué día, más -- opcional
    # -- valores conocidos/planeados para ese día que la API no puede inventar
    # (precio, evento, snap; ver api/feature_builder.py::OVERRIDABLE_FIELDS). El
    # resto (~190 columnas de lags/rolling/encoding/calendario) se calcula solo a
    # partir de la historia real de esa serie en data/processed/.
    series_id: str
    date: datetime.date
    version: str | None = None
    overrides: dict[str, float | int | str | None] | None = None


class ForecastResponse(BaseModel):
    level_id: int
    series_id: str
    date: datetime.date
    grain: str
    target: str
    version: str
    prediction: float


class LevelInfo(BaseModel):
    level_id: int  # config.Level.id (1-12, ver config/levels.py), lo que recibe /predict/{level_id}
    level: str  # artifact["level"] = level_str completo: "level_{level_id:02d}_{grain}_{name}"
    # (p.ej. "level_09_daily_store_dept"), no confundir con Level.name ("store_dept",
    # sin id/grain) ni con artifact["level_label"] ("level_09_daily", sin name).
    grain: str  # "daily"|"weekly" -- una fila por cada grain con artifact disponible
    target: str
    version: str | None
    wape_test: float | None
    wrmsse_test: float | None
