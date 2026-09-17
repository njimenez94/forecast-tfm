"""Reconstruye, para una serie y fecha puntuales, la fila de features que vería
el modelo en producción -- corriendo el mismo `src.features.pipeline.build_dataset()`
de entrenamiento sobre historia real (data/processed/), en vez de servir una fila
ya calculada de artifacts/datasets/ (lo que hace notebooks/05_api_negocio.ipynb hoy).

Ver api/main.py (POST /predict/{level_id}/forecast) y docs/informe.md §9
("Limitación vigente"): la API seguía sin reconstruir features desde datos
crudos -- esto lo resuelve para el caso "predecir día D de la serie S", con el
límite de que las exógenas del propio día D (precio/evento/snap) para fechas
fuera del histórico no se inventan: si no vienen en `overrides`, precio se
arrastra (carry-forward) y evento/snap se asumen "sin evento".

Por qué esto no hace falta el histórico completo por request para calcular
target-encoding cross-serie (src/features/encoding.py): se carga TODO el nivel
(filtrado a level.filters, igual que build_datasets.py) hasta `date < target_date`
más la única fila sintética de la serie pedida -- exactamente el mismo universo
que ve build_dataset() en training, solo que recortado a un día menos de
historia. Los niveles activos (1-9) son chicos (el más grande, level_09, ronda
135k filas), así que recorrerlos enteros por request es aceptable para el
alcance de este API de demo.
"""
import pandas as pd
import polars as pl

import config
from src.data.reader import read_parquet_pl
from src.features.pipeline import build_dataset

# Columnas exógenas del día objetivo que el caller puede pisar explícitamente.
# price_lag_N/price_change/price_vs_mean NO están acá: se derivan de avg_sell_price
# + historia real (igual que src/data/base_query.py::build_exog_query), no se
# inyectan sueltas para no dejarlas inconsistentes entre sí.
OVERRIDABLE_FIELDS = (
    "avg_sell_price", "snap", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
)

_PRICE_LAG_N = {"daily": 7, "weekly": 1}


def _level_filter_expr(level) -> pl.Expr | None:
    if not level.filters:
        return None
    return pl.all_horizontal([pl.col(c).is_in(v) for c, v in level.filters.items()])


def _calendar_fields(target_date: pd.Timestamp) -> dict:
    iso = target_date.isocalendar()
    dayofweek = int(iso.weekday)  # 1=lunes..7=domingo, igual que EXTRACT(ISODOW ...)
    return {
        "year": target_date.year,
        "month": target_date.month,
        "day": target_date.day,
        "dayofweek": dayofweek,
        "weekofyear": int(iso.week),
        "is_weekend": 1 if dayofweek >= 6 else 0,
    }


def _price_fields(own_history: pl.DataFrame, grain: str, avg_sell_price: float | None) -> dict:
    """price_lag_N/price_change/price_vs_mean del día objetivo, replicando
    src/data/base_query.py::build_exog_query (LAG por N *filas* -- no por N días
    calendario -- y media expandiendo excluyendo el propio día)."""
    price_lag_col = config.PRICE_LAG_COL[grain]
    n = _PRICE_LAG_N[grain]

    prices = own_history.sort("date")["avg_sell_price"]
    price_lag = prices[-n] if len(prices) >= n else None
    pmean = prices.mean() if len(prices) > 0 else None

    price_change = (
        avg_sell_price / price_lag - 1 if avg_sell_price is not None and price_lag not in (None, 0) else None
    )
    price_vs_mean = (
        avg_sell_price / pmean if avg_sell_price is not None and pmean not in (None, 0) else None
    )
    return {price_lag_col: price_lag, "price_change": price_change, "price_vs_mean": price_vs_mean}


def build_target_row(level, grain: str, series_id: str, target_date, overrides: dict | None = None) -> pd.DataFrame:
    """Devuelve una fila (pd.DataFrame de 1 fila) con las ~190 columnas de
    features del artifact para (level, grain, series_id, target_date), calculadas
    sobre historia real. Lanza FileNotFoundError si no hay datos procesados para
    ese nivel/grain, KeyError si la serie no existe o no tiene historia previa a
    target_date."""
    path = config.dataset_level_path(level, grain)
    if not path.exists():
        raise FileNotFoundError(f"No hay datos procesados para nivel {level.id} (grain={grain}) en {path}")

    target_date = pd.Timestamp(target_date)
    date_filter = pl.col("date") < target_date
    level_filter = _level_filter_expr(level)
    filter_expr = date_filter if level_filter is None else (date_filter & level_filter)
    history = read_parquet_pl(path, filter_expr=filter_expr)

    own_history = history.filter(pl.col("series_id") == series_id).sort("date")
    if own_history.is_empty():
        raise KeyError(
            f"series_id {series_id!r} no tiene historia antes de {target_date.date()} "
            f"en nivel {level.id} (grain={grain}) -- ¿existe esa serie en este nivel?"
        )
    last_row = own_history.tail(1)

    overrides = dict(overrides or {})
    unknown = set(overrides) - set(OVERRIDABLE_FIELDS)
    if unknown:
        raise ValueError(f"overrides no soportados: {sorted(unknown)} (permitidos: {list(OVERRIDABLE_FIELDS)})")

    static_dims = [d for d in level.dims if d not in config.EXCLUDE_AS_STATIC]
    # sales=0.0 (no None): mlforecast.preprocess() exige la columna target sin nulos
    # en toda la serie. El placeholder es inofensivo -- todo lo que deriva de "sales"
    # (lags/rolling/encoding/intermittency) usa shift(1)/ventanas que excluyen la
    # fila actual por diseño (ver src/features/*.py), así que el valor de esta fila
    # nunca se lee, solo hace falta que exista y sea numérico.
    row = {"series_id": series_id, "date": target_date, "sales": 0.0, "gross_sales": None}
    for col in static_dims:
        row[col] = last_row[col].item()

    avg_sell_price = overrides.get("avg_sell_price", last_row["avg_sell_price"].item())
    row["avg_sell_price"] = avg_sell_price
    row.update(_price_fields(own_history, grain, avg_sell_price))

    row["snap"] = overrides.get("snap", 0)
    for col in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        row[col] = overrides.get(col)
    row["has_event"] = 1 if row["event_name_1"] is not None else 0
    row["has_event_2"] = 1 if row["event_name_2"] is not None else 0
    row.update(_calendar_fields(target_date))

    missing = set(history.columns) - set(row)
    for col in missing:
        row[col] = None

    target_pl = pl.DataFrame([row]).select(history.columns).cast(history.schema)
    full = pl.concat([history, target_pl], how="vertical")

    featured = build_dataset(full, grain, static_dims)
    out = featured[(featured["series_id"] == series_id) & (featured["date"] == target_date)]
    if out.empty:
        raise RuntimeError("no se pudo reconstruir la fila objetivo (revisar series_id/overrides)")
    return out.tail(1)
