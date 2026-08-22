"""Modelos de referencia para comparar antes de invertir en feature selection y
tuning (ver notebook 02_model, sección "Comparación de modelos").

`naive_last_value` y `seasonal_naive` no entrenan nada: son el piso mínimo que
cualquier modelo debe superar. Los `fit_*` de ML (LightGBM, XGBoost, CatBoost,
HistGradientBoosting, Ridge) entrenan con hiperparámetros por defecto (sin
tuning) sobre la matriz de features (X_train/y_train). Los `fit_*` de
forecasting estadístico clásico (SARIMA, ETS, Theta, TBATS, Prophet) en cambio
ajustan serie por serie sobre train/valid (sin features, solo la propia serie
temporal) vía `_forecast_by_series`.

Todos se comparan en validación vía `evaluate_predictions`
(`src.evaluation.metrics`) antes de elegir una familia y recién ahí iterar
(selección de features, Optuna, etc.).
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


def naive_last_value(train_df, valid_df, target_col, group_col="agg_id"):
    """Repite la última venta observada por serie (día anterior al inicio de validación)."""
    last_train_value = train_df.sort_values("date").groupby(group_col)[target_col].last()
    return valid_df[group_col].map(last_train_value).fillna(0.0).to_numpy()


def seasonal_naive(train_df, valid_df, target_col, group_col="agg_id", season_length=7):
    """Repite el valor observado el mismo día de la semana en la última semana de
    train (estacionalidad semanal, típica en ventas diarias de retail)."""
    last_season = (
        train_df.sort_values("date")
        .groupby(group_col)[target_col]
        .apply(lambda s: s.to_numpy()[-season_length:])
    )

    valid_sorted = valid_df.sort_values([group_col, "date"])
    step_in_season = valid_sorted.groupby(group_col).cumcount().to_numpy()

    y_pred_sorted = np.array([
        season[i % len(season)] if (season := last_season.get(gid)) is not None and len(season) > 0 else 0.0
        for gid, i in zip(valid_sorted[group_col], step_in_season)
    ])

    return (
        pd.Series(y_pred_sorted, index=valid_sorted.index)
        .reindex(valid_df.index)
        .to_numpy()
    )


def drift(train_df, valid_df, target_col, group_col="agg_id"):
    """Método de deriva (drift / random walk with drift): proyecta la recta que une
    el primer y último valor de train por serie, extrapolada linealmente hacia
    adelante (paso h=1,2,... dentro de validación)."""
    def _endpoints(s):
        y = s.to_numpy(dtype=float)
        n = len(y)
        slope = (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0
        return pd.Series({"last": y[-1], "slope": slope})

    stats = train_df.sort_values("date").groupby(group_col)[target_col].apply(_endpoints).unstack()

    valid_sorted = valid_df.sort_values([group_col, "date"])
    h = valid_sorted.groupby(group_col).cumcount().to_numpy() + 1

    last = valid_sorted[group_col].map(stats["last"]).to_numpy()
    slope = valid_sorted[group_col].map(stats["slope"]).to_numpy()
    y_pred_sorted = np.nan_to_num(last + h * slope, nan=0.0)

    return (
        pd.Series(y_pred_sorted, index=valid_sorted.index)
        .reindex(valid_df.index)
        .to_numpy()
    )


def historical_mean(train_df, valid_df, target_col, group_col="agg_id"):
    """Promedio histórico completo de train por serie (constante en todo el
    horizonte de validación); solo tiene sentido si la serie es ~estacionaria."""
    hist_mean = train_df.groupby(group_col)[target_col].mean()
    return valid_df[group_col].map(hist_mean).fillna(0.0).to_numpy()


def moving_average(train_df, valid_df, target_col, group_col="agg_id", window=7):
    """Promedio de los últimos `window` valores observados en train por serie
    (suaviza ruido reciente), repetido de forma constante en validación."""
    ma = (
        train_df.sort_values("date")
        .groupby(group_col)[target_col]
        .apply(lambda s: s.to_numpy(dtype=float)[-window:].mean())
    )
    return valid_df[group_col].map(ma).fillna(0.0).to_numpy()


def _forecast_by_series(train_df, valid_df, target_col, fit_predict, group_col="agg_id"):
    """Ajusta y pronostica serie por serie con `fit_predict(train_sub, future_dates) ->
    array` (usado por los modelos estadísticos clásicos: ARIMA/SARIMA, ETS, Theta,
    TBATS, Prophet), alineando el resultado con `valid_df` (mismo criterio de
    sort/reindex que `drift`/`seasonal_naive`). Si `fit_predict` falla en una serie
    (muy corta, degenerada, no converge, etc.) cae al último valor observado en
    train para esa serie, en vez de abortar todo el ajuste."""
    train_groups = dict(iter(train_df.sort_values("date").groupby(group_col, sort=False)))
    valid_sorted = valid_df.sort_values([group_col, "date"])

    n_failed = 0
    parts, idx = [], []
    for gid, sub in valid_sorted.groupby(group_col, sort=False):
        h = len(sub)
        train_sub = train_groups.get(gid)
        if train_sub is None or train_sub.empty:
            fc = np.zeros(h)
        else:
            try:
                fc = np.asarray(fit_predict(train_sub, sub["date"].to_numpy()), dtype=float)
                if fc.size != h:
                    raise ValueError(f"esperaba {h} valores, se obtuvieron {fc.size}")
            except Exception:
                n_failed += 1
                fc = np.full(h, float(train_sub[target_col].to_numpy(dtype=float)[-1]))
        parts.append(fc)
        idx.append(sub.index)

    if n_failed:
        logger.warning(f"{n_failed} serie(s) fallaron el ajuste; se usa el último valor de train como fallback.")

    y_pred_sorted = pd.Series(np.concatenate(parts), index=np.concatenate(idx))
    return y_pred_sorted.reindex(valid_df.index).to_numpy()

def fit_sarima(train_df, valid_df, target_col, group_col="agg_id",
                order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """ARIMA/SARIMA por serie vía `statsmodels.SARIMAX`, con orden fijo (sin
    búsqueda automática tipo `auto_arima`: sobre ~1500-2000 observaciones por
    serie, un stepwise search por AIC tarda minutos por serie y no es viable
    para 70 series). `seasonal_order=(1,1,1,7)` da SARIMA (estacionalidad
    semanal, típica en ventas diarias); pasar `seasonal_order=(0,0,0,0)` da
    ARIMA simple. El método más citado en la literatura académica de
    forecasting estadístico clásico."""
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    def _fit_predict(train_sub, future_dates):
        y = train_sub[target_col].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = SARIMAX(
                y, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=200, method="lbfgs")
        return model.forecast(len(future_dates))

    return _forecast_by_series(train_df, valid_df, target_col, _fit_predict, group_col)


def fit_ets(train_df, valid_df, target_col, group_col="agg_id", seasonal_periods=7):
    """ETS / Holt-Winters (suavizado exponencial con tendencia y estacionalidad)
    por serie, vía `statsmodels`. Rápido y robusto; base de muchos benchmarks
    (M4/M5). Componentes aditivos (no multiplicativos) porque las ventas pueden
    valer 0."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    def _fit_predict(train_sub, future_dates):
        y = train_sub[target_col].to_numpy(dtype=float)
        model = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit()
        return model.forecast(len(future_dates))

    return _forecast_by_series(train_df, valid_df, target_col, _fit_predict, group_col)


def fit_theta(train_df, valid_df, target_col, group_col="agg_id", period=7):
    """Theta method (Assimakopoulos & Nikolopoulos, 2000), ganador del M3: simple
    (descompone la serie en dos "theta lines" y las combina) pero muy competitivo;
    poco conocido fuera del ámbito de forecasting. Deseasonalización aditiva (no
    multiplicativa) por los ceros en ventas."""
    from statsmodels.tsa.forecasting.theta import ThetaModel

    def _fit_predict(train_sub, future_dates):
        y = train_sub[target_col].to_numpy(dtype=float)
        model = ThetaModel(y, period=period, deseasonalize=True, method="additive").fit()
        return model.forecast(len(future_dates)).to_numpy()

    return _forecast_by_series(train_df, valid_df, target_col, _fit_predict, group_col)


def fit_tbats(train_df, valid_df, target_col, group_col="agg_id", season_length=(7,)):
    """TBATS (De Livera, Hyndman & Snyder, 2011): extensión de ETS con Box-Cox,
    ARMA de residuos y estacionalidades múltiples (ej. semanal + anual), pensada
    para series con más de una estacionalidad. Usa la implementación nativa de
    `statsforecast` (no el paquete `tbats` de PyPI, que está sin mantenimiento y
    es incompatible con scikit-learn>=1.8). `season_length=(7, 365.25)` habilita
    la doble estacionalidad, a costa de más tiempo de ajuste por serie; por
    defecto solo semanal."""
    from statsforecast.models import TBATS

    def _fit_predict(train_sub, future_dates):
        y = train_sub[target_col].to_numpy(dtype=float)
        model = TBATS(season_length=list(season_length)).fit(y)
        return model.predict(len(future_dates))["mean"]

    return _forecast_by_series(train_df, valid_df, target_col, _fit_predict, group_col)


def fit_prophet(train_df, valid_df, target_col, group_col="agg_id",
                 weekly_seasonality=True, yearly_seasonality=True):
    """Prophet (Taylor & Letham, 2018, Meta): descompone tendencia + estacionalidad
    + holidays con un modelo aditivo, robusto a datos faltantes/outliers y fácil
    de tunear; popular en industria. Silencia los logs de `cmdstanpy` (el backend
    de Prophet), muy verbosos por defecto."""
    import logging

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    from prophet import Prophet

    def _fit_predict(train_sub, future_dates):
        prophet_df = train_sub[["date", target_col]].rename(columns={"date": "ds", target_col: "y"})
        model = Prophet(weekly_seasonality=weekly_seasonality, yearly_seasonality=yearly_seasonality)
        model.fit(prophet_df)
        future = pd.DataFrame({"ds": future_dates})
        return model.predict(future)["yhat"].to_numpy()

    return _forecast_by_series(train_df, valid_df, target_col, _fit_predict, group_col)


def fit_lightgbm(X_train, y_train, categorical_features, random_state=42):
    model = LGBMRegressor(objective="rmse", random_state=random_state, verbosity=-1)
    model.fit(X_train, y_train, categorical_feature=categorical_features)
    return model


def fit_xgboost(X_train, y_train, random_state=42):
    """Requiere que las columnas categóricas ya sean dtype 'category' (usa
    `enable_categorical=True`, sin necesidad de encodear a mano)."""
    model = XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist",
        enable_categorical=True,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def catboost_features(X, categorical_features):
    """CatBoost no soporta el dtype category de pandas con nulos: convierte las
    categóricas a string, con los NaN reemplazados por 'missing'. Aplicar por
    igual a train (antes de `fit_catboost`) y a valid/test (antes de `predict`)."""
    X_cb = X.copy()
    for col in categorical_features:
        X_cb[col] = X_cb[col].astype(str).fillna("missing")
    return X_cb


def fit_catboost(X_train, y_train, categorical_features, random_state=42):
    X_train_cb = catboost_features(X_train, categorical_features)
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_state=random_state,
        cat_features=categorical_features,
        verbose=False,
    )
    model.fit(X_train_cb, y_train)
    return model


def fit_histgb(X_train, y_train, random_state=42):
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        categorical_features="from_dtype",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def fit_ridge(X_train, y_train, numerical_features, random_state=42):
    """Modelo lineal de referencia: solo usa las features numéricas (no maneja
    categóricas de alta cardinalidad)."""
    model = make_pipeline(SimpleImputer(strategy="median"), Ridge(random_state=random_state))
    model.fit(X_train[numerical_features], y_train)
    return model
