"""Forecasting estadístico clásico (SARIMA, ETS, Theta, TBATS, Prophet): ajustan
serie por serie sobre train/valid (sin features, solo la propia serie temporal)
vía `_forecast_by_series`. Todos se comparan en validación vía
`evaluate_predictions` (`src.evaluation`) antes de elegir una familia."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger


def _fit_predict_one(gid, sub, train_sub, target_col, fit_predict):
    """Ajusta una única serie; usado por `_forecast_by_series` tanto en el
    camino secuencial como en cada worker de `joblib` cuando `n_jobs != 1`."""
    h = len(sub)
    if train_sub is None or train_sub.empty:
        return sub.index, np.zeros(h), False
    try:
        fc = np.asarray(fit_predict(train_sub, sub["date"].to_numpy()), dtype=float)
        if fc.size != h:
            raise ValueError(f"esperaba {h} valores, se obtuvieron {fc.size}")
    except Exception:
        fc = np.full(h, float(train_sub[target_col].to_numpy(dtype=float)[-1]))
        return sub.index, fc, True
    return sub.index, fc, False


def _forecast_by_series(train_df, valid_df, target_col, fit_predict, group_col="series_id", n_jobs=-1):
    """Ajusta y pronostica serie por serie con `fit_predict(train_sub, future_dates) ->
    array` (usado por los modelos estadísticos clásicos: ARIMA/SARIMA, ETS, Theta,
    TBATS, Prophet), alineando el resultado con `valid_df` (mismo criterio de
    sort/reindex que `drift`/`seasonal_naive`). Si `fit_predict` falla en una serie
    (muy corta, degenerada, no converge, etc.) cae al último valor observado en
    train para esa serie, en vez de abortar todo el ajuste.

    Cada serie se ajusta de forma independiente (no comparten estado), así que
    se paralelizan con `joblib` (`n_jobs=-1` usa todos los cores); con miles de
    series (ej. `level_10_weekly_item` tiene ~3000) el ajuste secuencial es el
    cuello de botella real, no el costo de ajustar una sola serie."""
    train_groups = dict(iter(train_df.sort_values("date").groupby(group_col, sort=False)))
    valid_sorted = valid_df.sort_values([group_col, "date"])
    groups = list(valid_sorted.groupby(group_col, sort=False))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_one)(gid, sub, train_groups.get(gid), target_col, fit_predict)
        for gid, sub in groups
    )

    n_failed = sum(failed for _, _, failed in results)
    if n_failed:
        logger.warning(f"{n_failed} serie(s) fallaron el ajuste; se usa el último valor de train como fallback.")

    idx = np.concatenate([r[0] for r in results])
    parts = np.concatenate([r[1] for r in results])
    y_pred_sorted = pd.Series(parts, index=idx)
    return y_pred_sorted.reindex(valid_df.index).to_numpy()


def fit_sarima(train_df, valid_df, target_col, group_col="series_id",
                order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """ARIMA/SARIMA por serie vía `statsforecast.models.ARIMA` (nixtla), con
    orden fijo (sin búsqueda automática tipo `auto_arima`: sobre ~1500-2000
    observaciones por serie, un stepwise search por AIC tarda minutos por
    serie y no es viable para 70 series). `seasonal_order=(1,1,1,7)` da SARIMA
    (estacionalidad semanal, típica en ventas diarias); pasar
    `seasonal_order=(0,0,0,0)` da ARIMA simple.

    Se usa `statsforecast` en vez de `statsmodels.SARIMAX`: el statespace de
    `SARIMAX` para estacionalidad larga (ej. `season_length=52` en series
    semanales) tiene una dimensión de estado que crece con el período
    estacional (~`max(p+s·P, q+s·Q+1)`), y el filtro de Kalman escala cúbico
    con eso — cientos de veces más lento (y numéricamente inestable) que la
    implementación de `statsforecast`, que no sufre ese blow-up."""
    from statsforecast.models import ARIMA

    p, d, q = order
    P, D, Q, s = seasonal_order

    def _fit_predict(train_sub, future_dates):
        y = train_sub[target_col].to_numpy(dtype=float)
        model = ARIMA(
            order=(p, d, q), seasonal_order=(P, D, Q), season_length=s or 1,
        ).fit(y)
        return model.predict(len(future_dates))["mean"]

    return _forecast_by_series(train_df, valid_df, target_col, _fit_predict, group_col)


def fit_ets(train_df, valid_df, target_col, group_col="series_id", seasonal_periods=7):
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


def fit_theta(train_df, valid_df, target_col, group_col="series_id", period=7):
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


def fit_tbats(train_df, valid_df, target_col, group_col="series_id", season_length=(7,)):
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


def fit_prophet(train_df, valid_df, target_col, group_col="series_id",
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
