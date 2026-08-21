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

# Eventos con cierre total de tienda (venta ~0 independientemente de la demanda
# subyacente). Verificado empíricamente sobre data/processed: avg(sales | event)
# de Christmas es 0.22 (vs. 492.58 global, n=350), 3 órdenes de magnitud por debajo
# de cualquier otro evento (el segundo más bajo, Thanksgiving, promedia 301.86 --
# reduce demanda pero no cierra). Se modela aparte de event_name_1/event_type_1
# porque el efecto es un veto multiplicativo sobre la venta, no un desplazamiento
# aditivo.
_CLOSURE_EVENTS = {"Christmas"}

# is_store_closed exige, además del evento, que la venta observada ya sea marginal
# frente al nivel típico de esa serie (< 5% de su media). El evento por sí solo no
# basta como prueba de cierre en grains agregados (state/total), donde Christmas cae
# pero la suma de decenas de tiendas sigue siendo sustancial -- solo se fuerza a 0
# cuando el propio dato ya corrobora el cierre.
_CLOSURE_SALES_RATIO = 0.05

# Ventana (días) alrededor de un cierre en la que days_to_closure lleva valor; fuera
# de ella, NaN.
_CLOSURE_WINDOW = 7


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["agg_id", "date"])

    is_event = pl.col("has_event").cast(pl.Boolean) | pl.col("has_event_2").cast(pl.Boolean)
    event_date = pl.when(is_event).then(pl.col("date"))
    release_date = pl.when(pl.col("avg_sell_price").is_not_null()).then(pl.col("date"))
    running_max_price = pl.col("avg_sell_price").cum_max().over("agg_id")
    # is_in() sobre null devuelve null (lógica de Kleene), no false: sin fill_null
    # los días sin evento (event_name_1/2 null) quedarían como null en vez de 0.
    is_closure_event = (
        pl.col("event_name_1").is_in(_CLOSURE_EVENTS).fill_null(False)
        | pl.col("event_name_2").is_in(_CLOSURE_EVENTS).fill_null(False)
    )
    mean_sales = pl.col("sales").mean().over("agg_id")
    is_store_closed = (
        is_closure_event & (pl.col("sales") < _CLOSURE_SALES_RATIO * mean_sales)
    ).fill_null(False).cast(pl.Int8)

    # Distancia firmada al cierre más cercano (positiva = faltan N días, negativa =
    # pasaron N días, 0 el propio día de cierre), acotada a +-_CLOSURE_WINDOW y NaN
    # fuera de esa ventana. A diferencia de days_since_event/days_to_event (genéricos,
    # sin signo, uno por cada evento) esta es una única señal específica de cierres,
    # pensada para que el árbol distinga "día después del cierre" (rebote de demanda)
    # de un día normal sin depender de lag1, que en esos días vale 0 por el zero-out
    # de is_store_closed. Sin acotar, el rango real (+-180 días entre un Christmas y
    # el siguiente) diluye la señal entre cientos de valores similares en otras
    # temporadas y el árbol no la usa; NaN fuera de la ventana la deja casi siempre
    # nula salvo justo alrededor del cierre, mucho más fácil de explotar en un split.
    closure_date = pl.when(is_closure_event).then(pl.col("date"))
    days_since_closure = (pl.col("date") - closure_date.forward_fill().over("agg_id")) \
        .dt.total_days().cast(pl.Int32)
    days_to_closure = (closure_date.backward_fill().over("agg_id") - pl.col("date")) \
        .dt.total_days().cast(pl.Int32)
    signed_days_to_closure = (
        pl.when(days_to_closure.is_null()).then(-days_since_closure)
        .when(days_since_closure.is_null()).then(days_to_closure)
        .when(days_to_closure <= days_since_closure).then(days_to_closure)
        .otherwise(-days_since_closure)
    )
    signed_days_to_closure = (
        pl.when(signed_days_to_closure.abs() <= _CLOSURE_WINDOW)
        .then(signed_days_to_closure)
        .otherwise(None)
    )

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
        signed_days_to_closure.alias("days_to_closure"),
        pl.col("event_type_1").replace_strict(_EVENT_TYPE_CODES, default=0, return_dtype=pl.Int8)
            .alias("event_type_1_enc"),
        pl.col("event_type_2").replace_strict(_EVENT_TYPE_CODES, default=0, return_dtype=pl.Int8)
            .alias("event_type_2_enc"),
        pl.col("date").dt.quarter().cast(pl.Int8).alias("quarter"),
        (pl.col("date").dt.day() == 1).cast(pl.Int8).alias("is_month_start"),
        (pl.col("date") == pl.col("date").dt.month_end()).cast(pl.Int8).alias("is_month_end"),
        is_store_closed.alias("is_store_closed"),
    ).with_columns(
        pl.when(is_store_closed.cast(pl.Boolean)).then(0.0).otherwise(pl.col("sales"))
            .alias("sales"),
        pl.when(is_store_closed.cast(pl.Boolean)).then(0.0).otherwise(pl.col("gross_sales"))
            .alias("gross_sales"),
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
