"""Features derivadas sobre el dataset base (data/processed/) para dejarlo listo
para modelos (artifacts/datasets/, vía scripts/build_datasets.py). Toma lo que ya
calculó el SQL (precio, calendario básico, eventos crudos) y añade señales
adicionales que solo se pueden calcular con historia completa de la serie
(release, distancia a eventos, posición del precio, lags/rolling/momentum),
componiendo src.features.calendar/events/price/encoding/intermittency.
"""
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


__all__ = ["add_features", "add_lag_features"]
