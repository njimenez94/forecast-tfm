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
    level_id: int  # config.Level.id (1-12, ver config/levels.py), lo que recibe /predict/{level_id}
    level: str  # artifact["level"] = level_str completo: "level_{level_id:02d}_{grain}_{name}"
    # (p.ej. "level_09_daily_store_dept"), no confundir con Level.name ("store_dept",
    # sin id/grain) ni con artifact["level_label"] ("level_09_daily", sin name) --
    # ninguno de los dos se expone acá. Ojo: `grain` no es un parámetro de la API;
    # va empotrado en este string y en la práctica solo se ve un grain por
    # level_id+target (ver api/registry.py::_registry_key).
    target: str
    version: str | None
    wape_test: float | None
    wrmsse_test: float | None
