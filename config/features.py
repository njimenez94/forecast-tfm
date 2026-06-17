"""Configuración de features para mlforecast."""
from mlforecast.lag_transforms import RollingMean, RollingMax

TARGET = "sales"

# Ventana de entrenamiento en días hacia atrás desde el fin del dataset (None = máximo)
TRAIN_WINDOW_DAYS = {
    "daily":  [500, 730, 1095, None], # ~1.4y, 2y, 3y, max
    "weekly": [1095, 1460, None],      # 3y, 4y, max
}

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

# Lags que mlforecast genera automáticamente (en unidades de la frecuencia)
# 365 incluido para que cum365 tenga al menos un lag seguro (k >= N=365)
MLFORECAST_LAGS = {
    "daily": [7, 14, 21, 28, 35, 42, 56, 91, 182, 364],
    "weekly": [1, 2, 4, 8, 13, 17, 22, 26, 39, 52],
}

# Transforms sobre el lag base (base_shift=28d → aplicados sobre lag 28; sin leakage)
MLFORECAST_LAG_TRANSFORMS = {
    "daily":  {28: [RollingMean(28), RollingMax(28), RollingMean(91), RollingMax(91)]},
    "weekly": {1:  [RollingMean(4),  RollingMax(4),  RollingMean(13), RollingMax(13)]},
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
EXOG_COLS = [
    "avg_sell_price", "price_change", "price_vs_mean",
    "has_event", "has_event_2", "snap",
    "year", "month", "day", "dayofweek", "weekofyear", "is_weekend",
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
