"""Reportes de alto nivel: comparación de modelos y detalle de error por
serie/fila, compuestos a partir de src.evaluation.point/scaled."""
import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.point import bias, mae, rmse, rmsle, smape, tracking_signal, wape
from src.evaluation.scaled import (
    _SERIES_COL, clip_closed_stores, compute_mae_scales, compute_mase, compute_naive_scales,
    compute_spec, compute_wrmsse, spec,
)


def evaluate_predictions(train_df, valid_df, y_valid, y_pred_valid, name, fit_time=None, category=None,
                          stage=None, objective_score=None, target_col="sales", m=1):
    """WAPE/WRMSSE de un modelo sobre validación (clipando cierres conocidos).
    Pensada para acumular en una lista y comparar modelos, p.ej.:
    `model_results.append(evaluate_predictions(train, valid, y_valid, model.predict(X_valid), "LightGBM"))`

    `fit_time` (segundos, opcional) permite comparar también el costo de
    entrenamiento de cada modelo.
    `category` (opcional) permite etiquetar el modelo (p.ej. "Naive", "ML")
    para filtrar/comparar resultados por familia.
    `stage` (opcional) etiqueta en qué fase del pipeline se generó este resultado
    (p.ej. "bench", "post_feature_selection", "final" -- ver scripts/train_dataset.py).
    `objective_score` (opcional) es el valor ya calculado de `objective_metric`
    (rmse o deviance de Tweedie, según el objective del nivel) para este modelo --
    lo decide el caller (conoce cfg.objective), acá solo se guarda tal cual para
    poder comparar modelos por la métrica decisional real, coherente entre niveles
    rmse y tweedie (a diferencia de comparar directamente por `rmse` o `wrmsse`).
    `target_col`/`m`: columna objetivo real ('sales' o 'cumN') y su paso de naive
    scale -- ver compute_naive_scales. Default 'sales'/1: sin cambios de comportamiento.
    """
    y_pred_valid = clip_closed_stores(valid_df, y_pred_valid)
    result = {
        "model": name,
        "category": category,
        "stage": stage,
        "objective_score": objective_score,
        "wape": float(wape(y_valid, y_pred_valid)),
        "wrmsse": compute_wrmsse(train_df, valid_df, y_pred_valid, target_col=target_col, m=m),
        "mae": mae(y_valid, y_pred_valid),
        "rmse": rmse(y_valid, y_pred_valid),
        "smape": float(smape(y_valid, y_pred_valid)),
        "bias": float(bias(y_valid, y_pred_valid)),
        "rmsle": rmsle(y_valid, y_pred_valid),
        "tracking_signal": tracking_signal(y_valid, y_pred_valid),
        "spec": compute_spec(train_df, valid_df, y_pred_valid, target_col=target_col),
        "mase": compute_mase(train_df, valid_df, y_pred_valid, m=m, target_col=target_col),
        "fit_time": fit_time,
    }
    label = f"{name} [{category}]" if category else name
    msg = (
        f"{label:>25s} | WAPE: {result['wape']:.2%} | WRMSSE: {result['wrmsse']:.4f} "
    )
    if fit_time is not None:
        msg += f" | fit: {fit_time:.2f}s"
    logger.info(msg)
    return result


def build_predictions_report(train_df, eval_df, y_true, y_pred, target_col="sales",
                              id_cols=(_SERIES_COL, "date"), extra_cols=("gross_sales", "gross_sales_pred"), m=1):
    """Clipa cierres, calcula WAPE/WRMSSE globales y arma el detalle de error por fila.

    Pensada para reusarse en cualquier etapa (modelo simple, post-Optuna, modelo
    final): recibe el modelo ya evaluado (y_pred) en vez de reentrenar.
    """
    y_pred = clip_closed_stores(eval_df, y_pred)

    metrics = {
        "wape": float(wape(y_true, y_pred)),
        "wrmsse": compute_wrmsse(train_df, eval_df, y_pred, target_col=target_col, m=m),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": float(smape(y_true, y_pred)),
        "bias": float(bias(y_true, y_pred)),
        "rmsle": rmsle(y_true, y_pred),
        "tracking_signal": tracking_signal(y_true, y_pred),
        "spec": compute_spec(train_df, eval_df, y_pred, target_col=target_col),
        "mase": compute_mase(train_df, eval_df, y_pred, m=m, target_col=target_col),
    }

    df_pred = eval_df.copy()
    df_pred["y_pred"] = np.round(y_pred, 0).astype(float)
    df_pred["error"] = df_pred[target_col] - df_pred["y_pred"]
    df_pred["abs_error"] = df_pred["error"].abs()
    df_pred["wape"] = df_pred["abs_error"] / df_pred[target_col].abs()
    df_pred["bias"] = -df_pred["error"] / df_pred[target_col].abs()
    
    df_pred['gross_sales_pred'] = df_pred['y_pred'] * df_pred['avg_sell_price']

    cols = list(id_cols) + [target_col] + list(extra_cols) + ["y_pred", "error", "abs_error", "wape", "bias"]
    df_pred = df_pred[cols].sort_values(['series_id','date'])

    return metrics, df_pred


def build_series_metrics(train_df, df_pred, target_col="sales", group_col=_SERIES_COL,
                          weight_level=("date",), m=1):
    """WAPE/bias/WRMSSE por serie a partir del detalle de predicciones (salida de
    `build_predictions_report`, debe traer las columnas `y_pred`, `error`,
    `abs_error` y `gross_sales`). Incluye también las sumas por serie de
    `y_pred`/`error`/`abs_error` para poder inspeccionar la predicción cruda,
    no solo las métricas normalizadas; y el conteo de días medidos (`n_days`)
    y días con venta > 0 (`n_days_with_sales`) por serie, útil para distinguir
    series intermitentes de series con demanda continua.

    `weight_level` define cómo se pondera el error dentro del WRMSSE de cada serie:
    (group_col,) -> sin ponderación real (peso constante = total de la serie, igual a RMSSE)
    ("date",) -> pondera cada fecha por su gross_sales (días de más venta pesan más)
    (group_col, "date") -> pondera cada fila por su propio gross_sales
    """
    weight_level = list(weight_level)
    scales = compute_naive_scales(train_df, group_col, target_col=target_col, m=m)
    mae_scales = compute_mae_scales(train_df, group_col, m=m, target_col=target_col)

    def _wrmsse_for_group(g):
        scale = scales.get(g.name)
        if not scale:
            return np.nan
        weights = g.groupby(weight_level)["gross_sales"].transform("sum").to_numpy()
        sq_err = (g[target_col].to_numpy() - g["y_pred"].to_numpy()) ** 2
        if weights.sum() == 0:
            rmse = np.sqrt(np.mean(sq_err))
        else:
            rmse = np.sqrt(np.average(sq_err, weights=weights))
        return float(rmse / np.sqrt(scale))

    def _mase_for_group(g):
        scale = mae_scales.get(g.name)
        if not scale:
            return np.nan
        mae_i = np.mean(np.abs(g[target_col].to_numpy() - g["y_pred"].to_numpy()))
        return float(mae_i / scale)

    def _spec_for_group(g):
        g = g.sort_values("date")
        return spec(g[target_col].to_numpy(), g["y_pred"].to_numpy())

    by_series = df_pred.groupby(group_col)
    gross_sales_by_series = by_series["gross_sales"].sum()

    df_metrics = pd.DataFrame({
        "n_days": by_series.size(),
        "n_days_with_sales": by_series[target_col].apply(lambda s: int((s > 0).sum())),
        "sales": by_series[target_col].sum(),
        "y_pred": by_series["y_pred"].sum(),
        "error": by_series["error"].sum(),
        "abs_error": by_series["abs_error"].sum(),
        "gross_sales": gross_sales_by_series,
        "gross_sales_pct": gross_sales_by_series / df_pred["gross_sales"].sum(),
        "wape": by_series.apply(lambda g: wape(g[target_col], g["y_pred"])),
        "bias": by_series.apply(lambda g: bias(g[target_col], g["y_pred"])),
        "rmsle": by_series.apply(lambda g: rmsle(g[target_col], g["y_pred"])),
        "tracking_signal": by_series.apply(lambda g: tracking_signal(g[target_col], g["y_pred"])),
        "wrmsse": by_series.apply(_wrmsse_for_group),
        "mase": by_series.apply(_mase_for_group),
        "spec": by_series.apply(_spec_for_group),
    })

    return df_metrics.sort_values("gross_sales", ascending=False)


def build_all_series_metrics(train_df, valid_df, y_true, predictions, target_col="sales",
                              group_col=_SERIES_COL, weight_level=("date",),
                              id_cols=(_SERIES_COL, "date"), extra_cols=("gross_sales",), m=1):
    """`build_series_metrics` (WAPE/bias/WRMSSE/MASE/SPEC por serie) para varios
    modelos a la vez, concatenados en un único DataFrame con columna `model`.

    Pensada para comparar todos los modelos de `model_results` (no solo el
    modelo final) a nivel de serie, con las mismas métricas.

    `predictions`: dict {nombre_modelo: y_pred_valid} (mismo `y_true`/`valid_df`
    para todos, p.ej. el set de validación usado en `evaluate_predictions`).
    """
    frames = []
    for name, y_pred in predictions.items():
        _, df_pred = build_predictions_report(
            train_df, valid_df, y_true, y_pred, target_col=target_col,
            id_cols=id_cols, extra_cols=extra_cols, m=m,
        )
        df_metrics = build_series_metrics(
            train_df, df_pred, target_col=target_col, group_col=group_col,
            weight_level=weight_level, m=m,
        )
        frames.append(df_metrics.reset_index().assign(model=name))

    return pd.concat(frames, ignore_index=True)
