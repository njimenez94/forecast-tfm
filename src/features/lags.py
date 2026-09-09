"""Materialización de lags/rolling/momentum/expanding/seasonal-rolling."""
import warnings

import pandas as pd
from mlforecast import MLForecast

import config


def _build_lag_engine(grain: str) -> MLForecast:
    """MLForecast solo para preprocess(): sin modelos reales (models={}), porque
    preprocess() no toca self.models -- solo lo usan fit()/predict() -- así que no
    hace falta un LGBMRegressor real para reusar el motor de generación de lags."""
    return MLForecast(
        models={},
        freq=config.MLFORECAST_FREQ[grain],
        lags=config.MLFORECAST_LAGS[grain],
        lag_transforms=config.MLFORECAST_LAG_TRANSFORMS[grain],
        date_features=[],
    )


def add_lag_features(df: pd.DataFrame, grain: str, static_cols: list[str]) -> pd.DataFrame:
    """Materializa lags/rolling/momentum/expanding/seasonal (config.MLFORECAST_LAGS y
    _LAG_TRANSFORMS) como columnas, vía el mismo motor que usa mlforecast en
    entrenamiento (MLForecast.preprocess) -- resultado idéntico al que ve el modelo,
    sin reimplementar la lógica de lags. El recorte por horizonte de despliegue
    (lags menores al paso h de cada fila) se aplica después del split, en
    src.data.temporal_split.mask_horizon_leakage.

    dropna=False: conserva todas las filas (las de historia insuficiente quedan con
    NaN en lag/rolling, no se descartan). Necesario porque scripts/train_datasets.py
    lee este mismo parquet y vuelve a computar sus propios lags internamente sobre
    "sales" -- si aquí ya hubiéramos recortado filas, esa segunda pasada perdería
    historia dos veces. LightGBM maneja NaN nativamente; en un notebook, .dropna()
    si el modelo elegido no los soporta.
    """
    fcst = _build_lag_engine(grain)
    # utilsforecast inserta cada batch de columnas con df[names] = values -- con ~130
    # columnas de lag/rolling eso fragmenta el DataFrame y pandas lo advierte por cada
    # batch. Cosmético (no afecta el resultado), silenciado solo acá.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        return fcst.preprocess(
            df, id_col="series_id", time_col="date", target_col="sales", static_features=static_cols,
            dropna=False,
        )
