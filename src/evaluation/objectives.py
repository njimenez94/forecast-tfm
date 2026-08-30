"""Objetivo custom y factories de eval-metric de LightGBM."""
import numpy as np
from sklearn.metrics import mean_tweedie_deviance

from src.evaluation.point import rmse, wape
from src.evaluation.scaled import _PRICE_COL, _SERIES_COL, compute_naive_scales, compute_weights, compute_wrmsse


def objective_metric(y_true, y_pred, objective="rmse", tweedie_variance_power=1.5):
    """Métrica coherente con el `objective` de LightGBM del nivel (ver
    config.lgbm_params / TWEEDIE_LEVELS): rmse para el caso general, deviance
    de Tweedie para los niveles intermitentes. Es la métrica que debe decidir
    selección de features y ranking de Optuna -- WRMSSE se calcula aparte solo
    para informar, no para decidir."""
    if objective == "tweedie":
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
        return float(mean_tweedie_deviance(y_true, y_pred, power=tweedie_variance_power))
    return rmse(y_true, y_pred)


def wape_metric(y_true, y_pred):
    """Eval metric de LightGBM (name, score, is_higher_better) para WAPE."""
    return "wape", wape(y_true, y_pred), False


def _wrmsse_scales_weights(train_df, group_col=_SERIES_COL, price_col=_PRICE_COL, target_col="sales", m=1):
    """Precalcula scales/weights una sola vez para un `train_df` fijo: ambas
    son invariantes entre rondas de boosting (no dependen de las
    predicciones), así que recalcularlas en cada llamada del eval metric
    (una por ronda) es trabajo repetido sobre el mismo resultado."""
    scales = compute_naive_scales(train_df, group_col, target_col=target_col, m=m)
    weights = compute_weights(train_df, price_col, group_col) if price_col in train_df.columns else None
    return scales, weights


def make_wrmsse_metric(train_df, valid_df, target_col="sales", m=1):
    """Fábrica de eval metric de LightGBM para WRMSSE: cierra sobre train/valid ya
    que la callback de lgb sólo recibe (y_true, y_pred). Para el wrapper sklearn
    (`LGBMRegressor.fit(eval_metric=...)`).

    `target_col`/`m`: columna objetivo real ('sales' o 'cumN') y su paso de naive
    scale (1 para 'sales', N para cumN) -- ver compute_naive_scales."""
    scales, weights = _wrmsse_scales_weights(train_df, target_col=target_col, m=m)

    def _wrmsse_metric(y_true, y_pred):
        return "wrmsse", compute_wrmsse(train_df, valid_df, y_pred, scales=scales, weights=weights, target_col=target_col), False
    return _wrmsse_metric


def make_wrmsse_feval(train_df, valid_df, target_col="sales", m=1):
    """Igual que `make_wrmsse_metric` pero con la firma `(preds, eval_data)` que
    espera `feval` en la API nativa (`lgb.train`) en vez de `(y_true, y_pred)`."""
    scales, weights = _wrmsse_scales_weights(train_df, target_col=target_col, m=m)

    def _feval(preds, eval_data):
        return "wrmsse", compute_wrmsse(train_df, valid_df, preds, scales=scales, weights=weights, target_col=target_col), False
    return _feval
