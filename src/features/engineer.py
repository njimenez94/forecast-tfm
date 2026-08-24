"""Features derivadas sobre el dataset base (data/processed/) para dejarlo listo
para modelos (artifacts/datasets/, vía scripts/build_datasets.py). Toma lo que ya
calculó el SQL (precio, calendario básico, eventos crudos) y añade señales
adicionales que solo se pueden calcular con historia completa de la serie
(release, distancia a eventos, posición del precio, lags/rolling/momentum).
"""
import math
import warnings

import pandas as pd
import polars as pl

from src.modeling.train import build_fcst

_EVENT_TYPE_CODES = {"Sporting": 1, "Cultural": 2, "National": 3, "Religious": 4}

# Eventos sin cierre de tienda pero con caída de demanda clara y consistente,
# verificada empíricamente sobre data/processed/level_09_daily_store_dept.parquet:
# avg(sales | evento) de Thanksgiving es 301.86 (-39% vs 492.58 global, n=350) y de
# NewYear 368.12 (-25%). El resto de los ~30 eventos del calendario cae dentro de
# +-20% del promedio global, indistinguible de ruido a este nivel de agregación, así
# que no se generaliza a los 30 (evita columnas dispersas sin señal real).
_HIGH_IMPACT_EVENTS = ("Thanksgiving", "NewYear")

# Ventana (días) de la aproximación de año calendario usada para las features
# cíclicas de día del año (365.25 amortigua el corrimiento por años bisiestos).
_DAYS_PER_YEAR = 365.25

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


def _event_distance_features(event_name: str) -> tuple[pl.Expr, pl.Expr]:
    """days_since_<evento>/days_to_<evento>: mismo patrón forward_fill/backward_fill
    que days_since_event/days_to_event, pero acotado a un único nombre de evento
    (event_name_1 o event_name_2), para las señales de _HIGH_IMPACT_EVENTS."""
    is_this_event = (
        (pl.col("event_name_1") == event_name) | (pl.col("event_name_2") == event_name)
    ).fill_null(False)
    event_date = pl.when(is_this_event).then(pl.col("date"))
    days_since = (pl.col("date") - event_date.forward_fill().over("agg_id")) \
        .dt.total_days().cast(pl.Int32)
    days_to = (event_date.backward_fill().over("agg_id") - pl.col("date")) \
        .dt.total_days().cast(pl.Int32)
    return days_since, days_to


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

    # Codificación cíclica del calendario (sin/cos): dow/month/day-of-year como
    # enteros lineales rompen la continuidad dic->ene o dom->lun; el par sin/cos
    # preserva la distancia real entre extremos del período.
    dow = pl.col("date").dt.weekday().cast(pl.Float64)  # 1=lunes..7=domingo
    month = pl.col("date").dt.month().cast(pl.Float64)
    doy = pl.col("date").dt.ordinal_day().cast(pl.Float64)
    dow_angle = 2 * math.pi * (dow - 1) / 7
    month_angle = 2 * math.pi * (month - 1) / 12
    doy_angle = 2 * math.pi * (doy - 1) / _DAYS_PER_YEAR

    # Volatilidad de precio: desviación estándar móvil de 90 días (nivel ya lo da
    # price_vs_max/price_vs_mean; esto captura si el precio ha estado fluctuando --
    # promociones frecuentes -- vs. estable).
    price_volatility = (
        pl.col("avg_sell_price").rolling_std(window_size=90, min_samples=2).over("agg_id")
    )

    high_impact_exprs = {}
    for event_name in _HIGH_IMPACT_EVENTS:
        days_since, days_to = _event_distance_features(event_name)
        key = event_name.lower()
        high_impact_exprs[f"days_since_{key}"] = days_since.alias(f"days_since_{key}")
        high_impact_exprs[f"days_to_{key}"] = days_to.alias(f"days_to_{key}")

    df = df.with_columns(
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
        dow_angle.sin().cast(pl.Float32).alias("dow_sin"),
        dow_angle.cos().cast(pl.Float32).alias("dow_cos"),
        month_angle.sin().cast(pl.Float32).alias("month_sin"),
        month_angle.cos().cast(pl.Float32).alias("month_cos"),
        doy_angle.sin().cast(pl.Float32).alias("doy_sin"),
        doy_angle.cos().cast(pl.Float32).alias("doy_cos"),
        price_volatility.cast(pl.Float32).alias("price_volatility"),
        *high_impact_exprs.values(),
    ).with_columns(
        pl.when(is_store_closed.cast(pl.Boolean)).then(0.0).otherwise(pl.col("sales"))
            .alias("sales"),
        pl.when(is_store_closed.cast(pl.Boolean)).then(0.0).otherwise(pl.col("gross_sales"))
            .alias("gross_sales"),
    )

    return add_intermittency_features(df)


# Umbral (días) de racha de venta cero, con la serie ya en periodo activo (precio no
# nulo) y sin evento de cierre, a partir del cual se marca como probable quiebre de
# stock en vez de falta de demanda genuina.
_STOCKOUT_STREAK_THRESHOLD = 7


def add_intermittency_features(df: pl.DataFrame) -> pl.DataFrame:
    """Señales de intermitencia de demanda (zero_streak, pct_zero, ADI, CV²) y un
    proxy de quiebre de stock. df ya está ordenado por (agg_id, date) y trae 'sales'
    final (post zero-out de is_store_closed) y 'days_since_release'.

    Todo se calcula sobre pl.col("sales").shift(1).over("agg_id") -- el historial
    hasta *ayer*, nunca incluyendo la venta del día actual -- para no leakear el
    target: sin el shift, zero_streak==0 revelaría trivialmente que sales[t] > 0.
    """
    sales_prev = pl.col("sales").shift(1).over("agg_id")
    is_zero_prev = (sales_prev == 0).cast(pl.Float32)

    # Fecha de la última venta positiva estrictamente anterior a la fila actual:
    # se calcula el forward_fill incluyendo el día de hoy y luego se desplaza el
    # resultado un puesto, en vez de desplazar sales antes del forward_fill, porque
    # así conviven en una sola expresión sin encadenar dos .over() anidados.
    nonzero_date_incl_today = pl.when(pl.col("sales") > 0).then(pl.col("date"))
    last_sale_date_prev = nonzero_date_incl_today.forward_fill().over("agg_id").shift(1).over("agg_id")
    days_since_last_sale = (pl.col("date") - last_sale_date_prev).dt.total_days().cast(pl.Int32)
    zero_streak = (days_since_last_sale - 1).clip(lower_bound=0)

    pct_zero_28 = is_zero_prev.rolling_mean(window_size=28, min_samples=1).over("agg_id")
    pct_zero_90 = is_zero_prev.rolling_mean(window_size=90, min_samples=1).over("agg_id")

    # ADI expandiendo: días transcurridos desde el release (hasta ayer) / nº de días
    # con venta positiva (hasta ayer). Cuanto mayor, más intermitente la serie.
    days_elapsed_prev = (pl.col("days_since_release") - 1).clip(lower_bound=0)
    nonzero_count_prev = (
        (pl.col("sales") > 0).cast(pl.Int32).cum_sum().over("agg_id").shift(1).over("agg_id")
    )
    adi_expanding = (
        (days_elapsed_prev / nonzero_count_prev)
        .replace([float("inf"), float("-inf")], None)
        .cast(pl.Float32)
    )

    # CV² (coef. de variación al cuadrado) de la venta en la ventana de 90 días
    # previos. Aproximación sobre todos los días de la ventana (no solo los de venta
    # positiva, que sería el CV² "puro" de tamaño de demanda de la literatura
    # Syntetos-Boylan-Croston) -- mucho más simple de calcular vía rolling_std/mean
    # nativos de polars y sigue distinguiendo series erráticas de suaves.
    sales_prev_std = pl.col("sales").shift(1).over("agg_id") \
        .rolling_std(window_size=90, min_samples=2).over("agg_id")
    sales_prev_mean = pl.col("sales").shift(1).over("agg_id") \
        .rolling_mean(window_size=90, min_samples=2).over("agg_id")
    cv2_90 = (
        ((sales_prev_std / sales_prev_mean) ** 2)
        .replace([float("inf"), float("-inf")], None)
        .cast(pl.Float32)
    )

    is_likely_stockout = (
        pl.col("avg_sell_price").is_not_null()
        & (zero_streak >= _STOCKOUT_STREAK_THRESHOLD)
        & ~pl.col("is_store_closed").cast(pl.Boolean)
    ).fill_null(False).cast(pl.Int8)

    return df.with_columns(
        zero_streak.alias("zero_streak"),
        pct_zero_28.alias("pct_zero_28"),
        pct_zero_90.alias("pct_zero_90"),
        adi_expanding.alias("adi_expanding"),
        cv2_90.alias("cv2_90"),
        is_likely_stockout.alias("is_likely_stockout"),
    )


def add_lag_features(df: pd.DataFrame, grain: str, static_cols: list[str]) -> pd.DataFrame:
    """Materializa lags/rolling/momentum/expanding/seasonal (config.MLFORECAST_LAGS y
    _LAG_TRANSFORMS) como columnas, vía el mismo motor que usa mlforecast en
    entrenamiento (MLForecast.preprocess) -- resultado idéntico al que ve el modelo,
    sin reimplementar la lógica de lags. Usa el set completo del target 'sales' (sin
    el recorte por leakage que aplica valid_lags/valid_lag_transforms a targets
    cumN): quien entrene un modelo directo contra un target cumN debe restringirse a
    config.valid_lags/valid_lag_transforms(grain, f"cum{N}") para evitar leakage --
    este dataset da el superset de columnas, no filtra por target. El recorte por
    horizonte de despliegue (lags menores al paso h de cada fila) se aplica después
    del split, en src.data.split.mask_horizon_leakage.

    dropna=False: conserva todas las filas (las de historia insuficiente quedan con
    NaN en lag/rolling, no se descartan). Necesario porque scripts/train_datasets.py
    lee este mismo parquet y vuelve a computar sus propios lags internamente sobre
    "sales" -- si aquí ya hubiéramos recortado filas, esa segunda pasada perdería
    historia dos veces. LightGBM maneja NaN nativamente; en un notebook, .dropna()
    si el modelo elegido no los soporta.
    """
    fcst = build_fcst(grain, target="sales")
    # utilsforecast inserta cada batch de columnas con df[names] = values -- con ~130
    # columnas de lag/rolling eso fragmenta el DataFrame y pandas lo advierte por cada
    # batch. Cosmético (no afecta el resultado), silenciado solo acá.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        return fcst.preprocess(
            df, id_col="agg_id", time_col="date", target_col="sales", static_features=static_cols,
            dropna=False,
        )
