"""Features de calendario: posición dentro del mes y codificación cíclica del
día de semana/mes/año."""
import math

import polars as pl

# Ventana (días) de la aproximación de año calendario usada para las features
# cíclicas de día del año (365.25 amortigua el corrimiento por años bisiestos).
_DAYS_PER_YEAR = 365.25


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """quarter/is_month_start/is_month_end y codificación cíclica (sin/cos) de
    dow/month/doy: como enteros lineales rompen la continuidad dic->ene o
    dom->lun; el par sin/cos preserva la distancia real entre extremos del
    período."""
    dow = pl.col("date").dt.weekday().cast(pl.Float64)  # 1=lunes..7=domingo
    month = pl.col("date").dt.month().cast(pl.Float64)
    doy = pl.col("date").dt.ordinal_day().cast(pl.Float64)
    dow_angle = 2 * math.pi * (dow - 1) / 7
    month_angle = 2 * math.pi * (month - 1) / 12
    doy_angle = 2 * math.pi * (doy - 1) / _DAYS_PER_YEAR

    return df.with_columns(
        pl.col("date").dt.quarter().cast(pl.Int8).alias("quarter"),
        (pl.col("date").dt.day() == 1).cast(pl.Int8).alias("is_month_start"),
        (pl.col("date") == pl.col("date").dt.month_end()).cast(pl.Int8).alias("is_month_end"),
        dow_angle.sin().cast(pl.Float32).alias("dow_sin"),
        dow_angle.cos().cast(pl.Float32).alias("dow_cos"),
        month_angle.sin().cast(pl.Float32).alias("month_sin"),
        month_angle.cos().cast(pl.Float32).alias("month_cos"),
        doy_angle.sin().cast(pl.Float32).alias("doy_sin"),
        doy_angle.cos().cast(pl.Float32).alias("doy_cos"),
    )
