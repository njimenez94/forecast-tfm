"""Features derivadas sobre el dataset base (data/processed/) para dejarlo listo
para modelos (artifacts/datasets/, vía scripts/build_datasets.py). Toma lo que ya
calculó el SQL (precio, calendario básico, eventos crudos) y añade señales
adicionales que solo se pueden calcular con historia completa de la serie
(release, distancia a eventos, posición del precio, lags/rolling/momentum).
"""
import re

import pandas as pd
import polars as pl

from src.modeling.train import build_fcst

_EVENT_TYPE_CODES = {"Sporting": 1, "Cultural": 2, "National": 3, "Religious": 4}

# Eventos con cierre total de tienda (venta ~0 independientemente de la demanda
# subyacente). Verificado empíricamente sobre data/processed: avg(sales | event)
# de Christmas es 0.22 (vs. 492.58 global, n=350), 3 órdenes de magnitud por debajo
# de cualquier otro evento (el segundo más bajo, Thanksgiving, promedia 301.86 --
# reduce demanda pero no cierra). Se modela aparte de event_name_1/event_type_1
# porque el efecto es un veto multiplicativo sobre la venta, no un desplazamiento
# aditivo: un árbol de decisión necesita un split dedicado y con máxima señal
# (no compartido con las ~30 categorías de event_name_1) para aislarlo del resto
# de eventos, que sí son aditivos.
_CLOSURE_EVENTS = {"Christmas"}


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["agg_id", "date"])

    is_event = pl.col("has_event").cast(pl.Boolean) | pl.col("has_event_2").cast(pl.Boolean)
    event_date = pl.when(is_event).then(pl.col("date"))
    release_date = pl.when(pl.col("avg_sell_price").is_not_null()).then(pl.col("date"))
    running_max_price = pl.col("avg_sell_price").cum_max().over("agg_id")
    # is_in() sobre null devuelve null (lógica de Kleene), no false: sin fill_null
    # los días sin evento (event_name_1/2 null) quedarían como null en vez de 0.
    is_store_closed = (
        pl.col("event_name_1").is_in(_CLOSURE_EVENTS).fill_null(False)
        | pl.col("event_name_2").is_in(_CLOSURE_EVENTS).fill_null(False)
    ).cast(pl.Int8)

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
        is_store_closed.alias("is_store_closed"),
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
    out = fcst.preprocess(
        df, id_col="agg_id", time_col="date", target_col="sales", static_features=static_cols,
        dropna=False,
    )
    return add_closure_interactions(out)


# Columnas de "nivel" (lags y medias, en escala de ventas) generadas por mlforecast
# -- se excluyen std/min/max (volatilidad, no nivel) y las ratio *_truediv_* (ya
# normalizadas, gatearlas a 0 destruiría la señal en vez de corregirla).
_LEVEL_COL_PATTERN = re.compile(
    r"^(lag\d+|rolling_mean_lag\d+_window_size\d+"
    r"|expanding_mean_lag\d+"
    r"|seasonal_rolling_mean_lag\d+_season_length\d+_window_size\d+)$"
)


def add_closure_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Interacción explícita cierre-de-tienda x nivel histórico: anula todas las
    features de nivel (lags y medias moviles) cuando is_store_closed=1, en vez de
    dejar que el árbol descubra ese cruce por su cuenta feature a feature. Con solo
    ~350 filas de Christmas en todo el dataset, el split is_store_closed pierde
    contra la señal masiva de cada lag/rolling individual sin esta ayuda explícita
    -- neutralizar solo lag1 (la señal más fuerte) no basta: lag7, lag14, rolling
    means, etc. siguen arrastrando la predicción hacia arriba.
    """
    if "is_store_closed" not in df.columns:
        return df
    is_open = 1 - df["is_store_closed"]
    level_cols = [c for c in df.columns if _LEVEL_COL_PATTERN.match(c)]
    for col in level_cols:
        df[f"{col}_if_open"] = df[col] * is_open
    return df
