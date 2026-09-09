"""Configuración de features para mlforecast."""
import operator

import pandas as pd
from mlforecast.lag_transforms import (
    Combine, ExpandingMean, RollingMax, RollingMean, RollingMin, RollingStd, SeasonalRollingMean,
    core_tfms,
)

TARGET = "sales"


def _stats(*windows: int) -> list:
    """Mean/std/min/max en cada window_size (volatilidad + nivel, no solo el promedio)."""
    tfms = []
    for w in windows:
        tfms += [RollingMean(w), RollingStd(w), RollingMin(w), RollingMax(w)]
    return tfms


class _Momentum(Combine):
    """Combine con nombre corto (momentum_lag{N}_{short}_{long}) en vez del autogenerado."""

    def _get_name(self, lag: int) -> str:
        return f"momentum_lag{lag}_{self.tfm1.window_size}_{self.tfm2.window_size}"


def _momentum(short: int, long: int) -> Combine:
    """Media móvil corta / larga: >1 acelerando, <1 desacelerando (tendencia)."""
    return _Momentum(RollingMean(short), RollingMean(long), operator.truediv)


class _SeasonalMean(SeasonalRollingMean):
    """SeasonalRollingMean con nombre corto (seasonal_mean_lag{N}_{season_length}_{window_size}).

    ponytail: _set_core_tfm() del padre busca la impl en coreforecast por
    self.__class__.__name__, así que subclasificar rompe ese lookup (no pasa con
    Combine, que no lo usa) -- hay que armar el _core_tfm a mano con el nombre real.
    """

    def _get_name(self, lag: int) -> str:
        return f"seasonal_mean_lag{lag}_{self.season_length}_{self.window_size}"

    def _set_core_tfm(self, lag: int) -> "_SeasonalMean":
        self._core_tfm = core_tfms.SeasonalRollingMean(
            lag=lag, season_length=self.season_length, window_size=self.window_size,
            min_samples=self.min_samples,
        )
        return self

HORIZON = {
    "daily":  [7, 14, 21, 28, 35, 42, 364],
    "weekly": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
}

SEASON_LENGTH = {"daily": 7, "weekly": 52}
SN_WINDOWS = {"daily": [1, 7, 28, 364], "weekly": [1, 4, 52]}
MA_WINDOWS = {"daily": [7, 14, 21, 28, 35], "weekly": [2, 3, 4, 6, 8]}
VALID_PERIODS = {"daily": 28, "weekly": 4}
TEST_PERIODS = {"daily": 28, "weekly": 4}
VALID_YEAR_BLOCKS = 13


def to_days(grain: str, periods: int) -> int:
    return periods if grain == "daily" else periods * 7


def valid_days(grain: str) -> int:
    return to_days(grain, VALID_PERIODS[grain])


def valid_year_days(grain: str) -> int:
    return to_days(grain, VALID_PERIODS[grain] * VALID_YEAR_BLOCKS)


def test_days(grain: str) -> int:
    return to_days(grain, TEST_PERIODS[grain])

SPLIT_DATES = {
    "train": {
        "daily":  {"end": pd.Timestamp("2015-04-26")},
        "weekly": {"end": pd.Timestamp("2015-04-20")},
    },
    "valid": {
        "daily":  {"start": pd.Timestamp("2015-04-27"), "end": pd.Timestamp("2016-04-24")},
        "weekly": {"start": pd.Timestamp("2015-04-27"), "end": pd.Timestamp("2016-04-18")},
    },
    "test": {
        "daily":  {"start": pd.Timestamp("2016-04-25"), "end": pd.Timestamp("2016-05-22")},
        "weekly": {"start": pd.Timestamp("2016-04-25"), "end": pd.Timestamp("2016-05-16")},
    },
}

# Lags que mlforecast genera automáticamente (en unidades de la frecuencia).
# daily: denso 1-28 (cada día del bloque de horizonte VALID_PERIODS/TEST_PERIODS
# tiene su propio lag puntual -- ver src.data.temporal_split.mask_horizon_leakage/block_origins,
# que revalida cada fila por su propio h dentro del bloque de 28), después salta a
# múltiplos gruesos (35, 42, 56, 91, 182, 364). 364 (no 365) para alinear
# día-de-semana a un año.
MLFORECAST_LAGS = {
    "daily": [*range(1, 29), 35, 42, 56, 91, 182, 364],
    "weekly": [1, 2, 3, 4, 8, 13, 17, 22, 26, 39, 52],
}

# Transforms por lag base. Cada anchor: mean/std/min/max en varias ventanas + momentum
# (corta/larga) donde aplica. 2-6 rellenan el hueco entre el anchor=1 (denso) y el
# anchor=7 (rolling ya existía):
# antes, una fila con h entre 2 y 6 se quedaba sin ningún rolling propio y usaba
# directamente el de lag7 (ver src.data.temporal_split.mask_horizon_leakage).
MLFORECAST_LAG_TRANSFORMS = {
    "daily": {
        1:   _stats(7, 14, 21, 35) + [_momentum(7, 35)],
        2:   _stats(7, 14),
        3:   _stats(7, 14),
        4:   _stats(7, 14),
        5:   _stats(7, 14),
        6:   _stats(7, 14),
        7:   _stats(7, 14),
        28:  _stats(7, 28, 91) + [_momentum(7, 28)],
        91:  _stats(28, 91) + [ExpandingMean()],
        364: _stats(28, 91) + [
            _SeasonalMean(season_length=7, window_size=8),
            _SeasonalMean(season_length=364, window_size=2),
        ],
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

# Frecuencia pandas por granularidad -- weekly usa semana ISO lunes-domingo (el "date"
# de cada fila es el lunes, ver src.data.base_query.build_base_query), no la semana
# retail wm_yr_wk de M5 (sábado-viernes).
MLFORECAST_FREQ = {"daily": "D", "weekly": "W-MON"}

# Columnas categóricas estáticas excluidas de static_features (son el índice de serie)
EXCLUDE_AS_STATIC = {"series_id"}

# Columna de precio lag por granularidad (coincide con build_exog_query)
PRICE_LAG_COL = {"daily": "price_lag_7", "weekly": "price_lag_1"}

# Features exógenas time-varying (pasadas al modelo y a predict X_df)
# Las que siguen a "price_vs_max" las añade src/features/ sobre el
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


