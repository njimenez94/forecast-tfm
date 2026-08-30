"""Features derivadas sobre el dataset base (data/processed/) para dejarlo listo
para modelos (artifacts/datasets/, vía scripts/build_datasets.py). Toma lo que ya
calculó el SQL (precio, calendario básico, eventos crudos) y añade señales
adicionales que solo se pueden calcular con historia completa de la serie
(release, distancia a eventos, posición del precio, lags/rolling/momentum),
componiendo src.features.calendar/events/price/encoding/intermittency.
"""
import pandas as pd
import polars as pl

from src.features.calendar import add_calendar_features
from src.features.encoding import add_encoding_features
from src.features.events import add_event_features
from src.features.intermittency import add_intermittency_features
from src.features.lags import add_lag_features
from src.features.price import add_price_features


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["series_id", "date"])
    df = add_calendar_features(df)
    df = add_price_features(df)
    df = add_event_features(df)
    df = add_encoding_features(df)
    return add_intermittency_features(df)


def build_dataset(df: pl.DataFrame, grain: str, static_cols: list[str]) -> pd.DataFrame:
    """add_features + add_lag_features + downcast a dtypes compactos: el dataset
    ML-ready final que consumen scripts/train_dataset.py y los notebooks (ver
    scripts/build_datasets.py). float64->float32 (los day-counts de calendar.py/
    events.py salen float64 al pasar por to_pandas() con nulls) y str->category
    (dims de baja cardinalidad repetidas en cada fila) recortan ~30% de RAM al leer
    el parquet, sin tocar cada caller."""
    final = add_features(df).to_pandas()
    final = add_lag_features(final, grain, static_cols)
    final["date"] = pd.to_datetime(final["date"])

    float_cols = final.select_dtypes("float64").columns
    str_cols = final.select_dtypes("str").columns
    final[float_cols] = final[float_cols].astype("float32")
    final[str_cols] = final[str_cols].astype("category")
    return final


__all__ = ["add_features", "add_lag_features", "build_dataset"]
