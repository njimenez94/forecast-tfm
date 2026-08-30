"""Señales de intermitencia de demanda: rachas de venta cero, ADI, CV² y un
proxy de quiebre de stock."""
import polars as pl

# Umbral (días) de racha de venta cero, con la serie ya en periodo activo (precio no
# nulo) y sin evento de cierre, a partir del cual se marca como probable quiebre de
# stock en vez de falta de demanda genuina.
_STOCKOUT_STREAK_THRESHOLD = 7


def add_intermittency_features(df: pl.DataFrame) -> pl.DataFrame:
    """Señales de intermitencia de demanda (zero_streak, pct_zero, ADI, CV²) y un
    proxy de quiebre de stock. df ya está ordenado por (series_id, date) y trae 'sales'
    final (post zero-out de is_store_closed) y 'days_since_release'.

    Todo se calcula sobre pl.col("sales").shift(1).over("series_id") -- el historial
    hasta *ayer*, nunca incluyendo la venta del día actual -- para no leakear el
    target: sin el shift, zero_streak==0 revelaría trivialmente que sales[t] > 0.
    """
    sales_prev = pl.col("sales").shift(1).over("series_id")
    is_zero_prev = (sales_prev == 0).cast(pl.Float32)

    # Fecha de la última venta positiva estrictamente anterior a la fila actual:
    # se calcula el forward_fill incluyendo el día de hoy y luego se desplaza el
    # resultado un puesto, en vez de desplazar sales antes del forward_fill, porque
    # así conviven en una sola expresión sin encadenar dos .over() anidados.
    nonzero_date_incl_today = pl.when(pl.col("sales") > 0).then(pl.col("date"))
    last_sale_date_prev = nonzero_date_incl_today.forward_fill().over("series_id").shift(1).over("series_id")
    days_since_last_sale = (pl.col("date") - last_sale_date_prev).dt.total_days().cast(pl.Int32)
    zero_streak = (days_since_last_sale - 1).clip(lower_bound=0)

    pct_zero_28 = is_zero_prev.rolling_mean(window_size=28, min_samples=1).over("series_id")
    pct_zero_90 = is_zero_prev.rolling_mean(window_size=90, min_samples=1).over("series_id")

    # ADI expandiendo: días transcurridos desde el release (hasta ayer) / nº de días
    # con venta positiva (hasta ayer). Cuanto mayor, más intermitente la serie.
    days_elapsed_prev = (pl.col("days_since_release") - 1).clip(lower_bound=0)
    nonzero_count_prev = (
        (pl.col("sales") > 0).cast(pl.Int32).cum_sum().over("series_id").shift(1).over("series_id")
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
    sales_prev_std = pl.col("sales").shift(1).over("series_id") \
        .rolling_std(window_size=90, min_samples=2).over("series_id")
    sales_prev_mean = pl.col("sales").shift(1).over("series_id") \
        .rolling_mean(window_size=90, min_samples=2).over("series_id")
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
