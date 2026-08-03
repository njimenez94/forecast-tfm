"""Features derivadas sobre el dataset base (data/processed/) para dejarlo listo
para modelos (artifacts/datasets/, vía scripts/build_datasets.py). Toma lo que ya
calculó el SQL (precio, calendario básico, eventos crudos) y añade señales
adicionales que solo se pueden calcular con historia completa de la serie
(release, distancia a eventos, posición del precio, lags/rolling/momentum).
"""
import pandas as pd
import polars as pl

from src.modeling.train import build_fcst

_EVENT_TYPE_CODES = {"Sporting": 1, "Cultural": 2, "National": 3, "Religious": 4}


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["agg_id", "date"])

    is_event = pl.col("has_event").cast(pl.Boolean) | pl.col("has_event_2").cast(pl.Boolean)
    event_date = pl.when(is_event).then(pl.col("date"))
    release_date = pl.when(pl.col("avg_sell_price").is_not_null()).then(pl.col("date"))
    running_max_price = pl.col("avg_sell_price").cum_max().over("agg_id")

    return df.with_columns(
        (pl.col("avg_sell_price") / running_max_price)
            .replace([float("inf"), float("-inf")], None)
            .cast(pl.Float32)
            .alias("price_vs_max"),
        (pl.col("date") - release_date.min().over("agg_id"))
            .dt.total_days().cast(pl.Int32).alias("days_since_release"),
        (pl.col("date") - event_date.forward_fill().over("agg_id"))
            .dt.total_days().cast(pl.Int32).alias("days_since_event"),
        (event_date.backward_fill().over("agg_id") - pl.col("date"))
            .dt.total_days().cast(pl.Int32).alias("days_to_event"),
        pl.col("event_type_1").replace_strict(_EVENT_TYPE_CODES, default=0, return_dtype=pl.Int8)
            .alias("event_type_1_enc"),
        pl.col("event_type_2").replace_strict(_EVENT_TYPE_CODES, default=0, return_dtype=pl.Int8)
            .alias("event_type_2_enc"),
        pl.col("date").dt.quarter().cast(pl.Int8).alias("quarter"),
        (pl.col("date").dt.day() == 1).cast(pl.Int8).alias("is_month_start"),
        (pl.col("date") == pl.col("date").dt.month_end()).cast(pl.Int8).alias("is_month_end"),
    )


def add_lag_features(df: pd.DataFrame, grain: str, static_cols: list[str]) -> pd.DataFrame:
    """Materializa lags/rolling/momentum/expanding/seasonal (config.MLFORECAST_LAGS y
    _LAG_TRANSFORMS) como columnas, vía el mismo motor que usa mlforecast en
    entrenamiento (MLForecast.preprocess) -- resultado idéntico al que ve el modelo,
    sin reimplementar la lógica de lags. Usa el set completo del target 'sales' (sin
    el recorte por leakage que aplica valid_lags/valid_lag_transforms a targets
    cumN): quien entrene un modelo directo contra un target cumN debe restringirse a
    config.valid_lags/valid_lag_transforms(grain, f"cum{N}") para evitar leakage --
    este dataset da el superset de columnas, no filtra por target.

    dropna=False: conserva todas las filas (las de historia insuficiente quedan con
    NaN en lag/rolling, no se descartan). Necesario porque scripts/train_datasets.py
    lee este mismo parquet y vuelve a computar sus propios lags internamente sobre
    "sales" -- si aquí ya hubiéramos recortado filas, esa segunda pasada perdería
    historia dos veces. LightGBM maneja NaN nativamente; en un notebook, .dropna()
    si el modelo elegido no los soporta.
    """
    fcst = build_fcst(grain, target="sales")
    return fcst.preprocess(
        df, id_col="agg_id", time_col="date", target_col="sales", static_features=static_cols,
        dropna=False,
    )
