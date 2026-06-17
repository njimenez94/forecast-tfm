import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast

import config


def build_fcst(grain: str, target: str = "sales") -> MLForecast:
    """Construye el MLForecast con lags seguros para el target dado.

    Para targets acumulados 'cumN', filtra lags < N para evitar leakage
    (lag_k(cumN)[t] incluye sales futuras si k < N).
    """
    return MLForecast(
        models={"lgb": lgb.LGBMRegressor(**config.LGBM_PARAMS)},
        freq=config.MLFORECAST_FREQ[grain],
        lags=config.valid_lags(grain, target),
        lag_transforms=config.valid_lag_transforms(grain, target),
        date_features=[],
    )


def encode_static(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).astype("category").cat.codes
    return df
