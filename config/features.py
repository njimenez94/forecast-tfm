"""Configuración de features para mlforecast."""
import operator

from mlforecast.lag_transforms import (
    Combine, ExpandingMean, RollingMax, RollingMean, RollingMin, RollingStd, SeasonalRollingMean,
)

TARGET = "sales"


def _stats(*windows: int) -> list:
    """Mean/std/min/max en cada window_size (volatilidad + nivel, no solo el promedio)."""
    tfms = []
    for w in windows:
        tfms += [RollingMean(w), RollingStd(w), RollingMin(w), RollingMax(w)]
    return tfms


def _momentum(short: int, long: int) -> Combine:
    """Media móvil corta / larga: >1 acelerando, <1 desacelerando (tendencia)."""
    return Combine(RollingMean(short), RollingMean(long), operator.truediv)

# Horizontes de validación por granularidad (días para daily, semanas para weekly)
HORIZON = {
    "daily":  [7, 14, 21, 28, 35, 42, 365],
    "weekly": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
}

# Horizontes acumulados a experimentar (días para daily, semanas para weekly)
# Genera targets cum7, cum14, ... donde cumN predice la suma de los próximos N períodos
CUM_HORIZONS = {
    "daily":  [7, 14, 21, 28, 35, 42, 365],
    "weekly": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
}

# Lags que mlforecast genera automáticamente (en unidades de la frecuencia).
# 1-3/1-2 cortos para autocorrelación inmediata (solo target 'sales': para cumN los
# filtra valid_lags por leakage). 364 (no 365) para alinear día-de-semana a un año.
MLFORECAST_LAGS = {
    "daily": [1, 2, 3, 7, 14, 21, 28, 35, 42, 56, 91, 182, 364],
    "weekly": [1, 2, 3, 4, 8, 13, 17, 22, 26, 39, 52],
}

# Transforms por lag base (shift → sin leakage si shift >= N del target cumN, ver
# valid_lag_transforms). Cada anchor: mean/std/min/max en varias ventanas + momentum
# (corta/larga) donde aplica. 365/52 son el único anchor seguro para cum365/cum52.
MLFORECAST_LAG_TRANSFORMS = {
    "daily": {
        7:   _stats(7, 14),
        28:  _stats(7, 28, 91) + [_momentum(7, 28)],
        91:  _stats(28, 91) + [ExpandingMean()],
        364: _stats(28, 91) + [SeasonalRollingMean(season_length=7, window_size=8)],
        365: [RollingMean(28), RollingMean(91)],
    },
    "weekly": {
        1:  _stats(4, 13),
        4:  _stats(4, 13, 26) + [_momentum(4, 13)],
        13: _stats(13, 26) + [ExpandingMean()],
        52: _stats(4, 13),
    },
}

# Features de calendario derivadas de ds por mlforecast
MLFORECAST_DATE_FEATURES = ["dayofweek", "month", "week"]

# Frecuencia pandas por granularidad
MLFORECAST_FREQ = {"daily": "D", "weekly": "W-SAT"}

# Columnas categóricas estáticas excluidas de static_features (son el índice de serie)
EXCLUDE_AS_STATIC = {"agg_id", "item_id"}

# Columna de precio lag por granularidad (coincide con build_exog_query)
PRICE_LAG_COL = {"daily": "price_lag_7", "weekly": "price_lag_1"}

# Features exógenas time-varying (pasadas al modelo y a predict X_df)
# Las que siguen a "price_vs_max" las añade src/features/engineer.py sobre el
# dataset base; solo existen en artifacts/datasets/ (no en data/processed/).
#
# zero_streak/pct_zero_28/pct_zero_90/adi_expanding/cv2_90/is_likely_stockout son
# cuasi-estáticas: dependen de sales histórica (hasta ayer, sin leakage), no de
# calendario/precio futuro-conocido. Hoy el pipeline de train_datasets.py solo hace
# backtest (predict sobre fechas históricas ya conocidas en valid/test, no forecast
# real a futuro), así que valid/test ya traen estas columnas calculadas
# correctamente desde el parquet -- no hace falta ningún tratamiento especial en
# src/data/split.py. Si en el futuro se agrega un script de forecast genuino más
# allá de la última fecha del dataset, ahí sí habría que congelar estas columnas al
# último valor conocido en vez de dejarlas NaN.
EXOG_COLS = [
    "avg_sell_price", "price_change", "price_vs_mean",
    "has_event", "has_event_2", "snap",
    "year", "month", "day", "dayofweek", "weekofyear", "is_weekend",
    "price_vs_max", "days_since_release",
    "event_type_1_enc", "event_type_2_enc", "days_since_event", "days_to_event",
    "quarter", "is_month_start", "is_month_end",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "price_volatility",
    "days_since_thanksgiving", "days_to_thanksgiving",
    "days_since_newyear", "days_to_newyear",
    "zero_streak", "pct_zero_28", "pct_zero_90", "adi_expanding", "cv2_90",
    "is_likely_stockout",
]


def cum_n(target: str) -> int | None:
    """Devuelve N para un target 'cumN', o None para 'sales'."""
    if target == "sales":
        return None
    return int(target[3:])


def valid_lags(grain: str, target: str) -> list[int]:
    """Lags seguros para el target dado.

    Para cumN: necesita lag k >= N para evitar leakage
    (lag_k(cumN)[t] = cumN[t-k] involucra sales hasta t-k+N; safe si t-k+N <= t, i.e. k>=N).
    Si ningún lag existente cumple, devuelve [N] como mínimo.
    """
    n = cum_n(target)
    all_lags = list(MLFORECAST_LAGS[grain])
    if n is None:
        return all_lags
    safe = [k for k in all_lags if k >= n]
    return safe if safe else [n]


def valid_lag_transforms(grain: str, target: str) -> dict:
    """Lag transforms seguros para el target dado (base_lag >= N para cumN)."""
    n = cum_n(target)
    if n is None:
        return dict(MLFORECAST_LAG_TRANSFORMS[grain])
    return {k: v for k, v in MLFORECAST_LAG_TRANSFORMS[grain].items() if k >= n}
