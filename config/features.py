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

# Horizontes de validación por granularidad (días para daily, semanas para weekly)
HORIZON = {
    "daily":  [7, 14, 21, 28, 35, 42, 364],
    "weekly": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
}

# Split temporal estandar: mismo valid/test para todos los notebooks/scripts que
# entrenen contra un nivel/grain dado. Unidad = periodos de la granularidad (días
# para daily, semanas para weekly); to_days() convierte a días de calendario, que
# es en lo que trabaja date_split.
SEASON_LENGTH = {"daily": 7, "weekly": 52}

# Ventanas para los baselines "seasonal naive" / "moving average" (ver
# scripts/train_dataset.py:run_baseline_naive).
SN_WINDOWS = {"daily": [1, 7, 28, 364], "weekly": [1, 4, 52]}
MA_WINDOWS = {"daily": [7, 14, 21, 28, 35], "weekly": [2, 3, 4, 6, 8]}
# Igual a TEST_PERIODS a propósito: es el horizonte real de despliegue, y también el
# tamaño de bloque que src.data.split.build_feature_matrices usa para revalidar
# horizonte en X_valid/X_test (ver block_origins). Con valid_days == valid_days(grain)
# (el default, un único bloque) esto da feature selection/tuning de Optuna sobre el
# mismo problema que después se evalúa en test. Un valid_days más largo (p.ej.
# valid_year_days) sigue revalidando por bloques de este tamaño en vez de un único
# origen para todo el split, así no enmascara de más la señal reciente.
VALID_PERIODS = {"daily": 28, "weekly": 4}
TEST_PERIODS = {"daily": 28, "weekly": 4}

# Cantidad de bloques de VALID_PERIODS que arma un valid "último año completo" (pasar
# a date_split(valid_days=...) -- build_feature_matrices ya revalida por bloques de
# VALID_PERIODS sin ningún parámetro extra, ver arriba): 13*28 = 364 días,
# 13*4 = 52 semanas, ambos ~1 año y múltiplo exacto del bloque.
VALID_YEAR_BLOCKS = 13


def to_days(grain: str, periods: int) -> int:
    return periods if grain == "daily" else periods * 7


def valid_days(grain: str) -> int:
    return to_days(grain, VALID_PERIODS[grain])


def valid_year_days(grain: str) -> int:
    return to_days(grain, VALID_PERIODS[grain] * VALID_YEAR_BLOCKS)


def test_days(grain: str) -> int:
    return to_days(grain, TEST_PERIODS[grain])


# Fechas fijas del split temporal estándar, a mano por grain (dataset M5, no cambia
# entre corridas -- src.data.split.split_data las usa directo en vez de derivarlas de
# df[date].max() en runtime). Las 5 son fechas REALES de alguna fila (ver calendario):
# *_START el primer día de ese split, *_END el último.
#
# weekly usa semana ISO lunes-domingo (no la semana retail wm_yr_wk de M5, sábado-
# viernes -- ver src.data.base_query.build_base_query), y cada fila queda fechada con
# su lunes. El último día real de venta, 2016-05-22, es justo un domingo: cierra la
# semana lunes 2016-05-16 a domingo 2016-05-22 con sus 7 días completos, así que
# TEST_END["weekly"] ancla a ese lunes sin necesidad de descartar ninguna semana
# parcial al final.
TRAIN_END = {
    "daily":  pd.Timestamp("2015-04-26"),
    "weekly": pd.Timestamp("2015-04-20"),
}
VALID_START = {
    "daily":  pd.Timestamp("2015-04-27"),
    "weekly": pd.Timestamp("2015-04-27"),
}
VALID_END = {
    "daily":  pd.Timestamp("2016-04-24"),
    "weekly": pd.Timestamp("2016-04-18"),
}
TEST_START = {
    "daily":  pd.Timestamp("2016-04-25"),
    "weekly": pd.Timestamp("2016-04-25"),
}
TEST_END = {
    "daily":  pd.Timestamp("2016-05-22"),
    "weekly": pd.Timestamp("2016-05-16"),
}

# Horizontes acumulados a experimentar (días para daily, semanas para weekly)
# Genera targets cum7, cum14, ... donde cumN predice la suma de los próximos N períodos
# (incluyendo el período actual, ver build_cum_query).
CUM_HORIZONS = {
    "daily":  [7, 14, 21, 28, 35, 42, 49, 56, 364],
    "weekly": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
}

# Objetivos cumN a evaluar/entrenar en esta ronda (en períodos de la granularidad:
# días para niveles daily, semanas para los weekly 10-12 -- mismos valores que
# CUM_HORIZONS, un subset). El máximo (56) también fija cuánto se reserva al final de
# cada serie en date_split (ver tail_reserve_days en scripts/train_dataset.py.split_data,
# convertido a días de calendario vía to_days()): cumN es NULL en los últimos N
# períodos de cada serie (ventana forward incompleta), y como el test/valid siempre
# cae en el final de la serie, reservar el máximo -- no el N propio de cada target --
# deja a cum7/14/21/28/35/42/49/56 evaluados sobre exactamente el mismo test/valid
# window, así las métricas entre objetivos son comparables entre sí.
CUM_EVAL_HORIZONS = [7, 14, 21, 28, 35, 42, 49, 56]

# Lags que mlforecast genera automáticamente (en unidades de la frecuencia).
# daily: denso 1-28 (cada día del bloque de horizonte VALID_PERIODS/TEST_PERIODS
# tiene su propio lag puntual -- ver src.data.split.mask_horizon_leakage/block_origins,
# que revalida cada fila por su propio h dentro del bloque de 28), después salta a
# múltiplos gruesos (35, 42, 56, 91, 182, 364). 364 (no 365) para alinear
# día-de-semana a un año. Filtro por leakage de targets cumN vía valid_lags.
MLFORECAST_LAGS = {
    "daily": [*range(1, 29), 35, 42, 56, 91, 182, 364],
    "weekly": [1, 2, 3, 4, 8, 13, 17, 22, 26, 39, 52],
}

# Transforms por lag base (shift → sin leakage si shift >= N del target cumN, ver
# valid_lag_transforms). Cada anchor: mean/std/min/max en varias ventanas + momentum
# (corta/larga) donde aplica. 364/52 son el único anchor seguro para cum364/cum52.
# 2-6 rellenan el hueco entre el anchor=1 (denso) y el anchor=7 (rolling ya existía):
# antes, una fila con h entre 2 y 6 se quedaba sin ningún rolling propio y usaba
# directamente el de lag7 (ver src.data.split.mask_horizon_leakage).
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

    Para 'sales' devuelve el set completo sin filtrar: el recorte por horizonte de
    despliegue (un lag k solo es seguro para las filas con h <= k) se aplica fila a
    fila en src.data.split.mask_horizon_leakage, después del split -- acá no se
    conoce todavía dónde cae valid_start/test_start.
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
