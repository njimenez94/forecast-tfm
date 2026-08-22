import numpy as np
import pandas as pd

_SERIES_COL = "agg_id"
_PRICE_COL = "avg_sell_price"


def wape(y_true, y_pred):
    """WAPE: error absoluto total como fracción de las ventas totales."""
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def bias(y_true, y_pred):
    """Bias del forecast; positivo = sobreestima, normalizado por ventas."""
    return (y_pred - y_true).sum() / np.abs(y_true).sum()


def smape(y_true, y_pred):
    """SMAPE: error porcentual simétrico promedio (con epsilon)."""
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return np.mean(2 * np.abs(y_pred - y_true) / denom)


def spec(y_true, y_pred, a1=0.75, a2=0.25):
    """Stock-keeping-oriented Prediction Error Costs (SPEC), vectorizada con
    sumas acumuladas (O(n^2) en vez de O(n^3)). Penaliza el desfase temporal
    entre demanda real y predicha, no solo la magnitud del error; pensada para
    demanda intermitente. https://arxiv.org/abs/2004.10537

    a1: peso de costos de oportunidad (under-stock). a2: peso de costos de
    mantener stock (over-stock). Requiere que `y_true`/`y_pred` estén
    ordenados por fecha (serie temporal de una sola serie, no datos apilados).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.size
    assert n > 0 and y_pred.size == n

    cy = np.concatenate(([0.0], np.cumsum(y_true)))
    cf = np.concatenate(([0.0], np.cumsum(y_pred)))

    delta1 = cy[1:][:, None] - cf[1:][None, :]
    delta2 = cf[1:][:, None] - cy[1:][None, :]

    term = np.maximum(0.0, np.maximum(
        a1 * np.minimum(y_true[:, None], delta1),
        a2 * np.minimum(y_pred[:, None], delta2),
    ))

    i_idx = np.arange(1, n + 1)[:, None]
    t_idx = np.arange(1, n + 1)[None, :]
    weight = (t_idx - i_idx + 1).astype(float)
    mask = i_idx <= t_idx

    return float(np.sum(term * weight * mask) / n)


def compute_scales(train_df, group_col=_SERIES_COL):
    """Denominador RMSSE por serie: MSE del naive one-step in-sample desde la
    primera venta no-cero (convención M5)."""
    scales = {}
    for gid, sub in train_df.sort_values("date").groupby(group_col, sort=False):
        y = sub["sales"].to_numpy(dtype=float)
        nz = np.flatnonzero(y)
        if nz.size == 0:
            continue
        y = y[nz[0]:]
        if y.size < 2:
            continue
        scales[gid] = float(np.mean(np.diff(y) ** 2))
    return scales


def compute_weights(train_df, price_col=_PRICE_COL, group_col=_SERIES_COL,
                    last_days=28):
    """Peso por serie = ventas en $ de los últimos `last_days` del train."""
    cutoff = train_df["date"].max() - pd.Timedelta(days=last_days)
    recent = train_df[train_df["date"] > cutoff]
    dollar = (recent["sales"] * recent[price_col]).groupby(recent[group_col]).sum()
    return dollar.to_dict()


def calculate_rmsse(valid_df, scales, group_col=_SERIES_COL):
    """RMSSE por serie. valid_df necesita: group_col, sales, forecast."""
    out = {}
    for gid, sub in valid_df.groupby(group_col, sort=False):
        scale = scales.get(gid)
        if not scale:
            continue
        num = np.mean((sub["sales"].to_numpy() - sub["forecast"].to_numpy()) ** 2)
        out[gid] = float(np.sqrt(num / scale))
    return out


def calculate_wrmsse(valid_df, scales, weights=None, group_col=_SERIES_COL):
    """WRMSSE = Σ w_i · RMSSE_i (pesos normalizados). weights=None → promedio."""
    rmsse = calculate_rmsse(valid_df, scales, group_col)
    if not rmsse:
        return float("nan")
    if weights is None:
        return float(np.mean(list(rmsse.values())))
    w = {g: weights.get(g, 0.0) for g in rmsse}
    total = sum(w.values())
    if total == 0:
        return float(np.mean(list(rmsse.values())))
    return float(sum(rmsse[g] * w[g] / total for g in rmsse))


def calculate_spec(valid_df, weights=None, group_col=_SERIES_COL, a1=0.75, a2=0.25):
    """SPEC por serie (necesita orden temporal), agregado como promedio ponderado
    igual que WRMSSE. valid_df necesita: group_col, date, sales, forecast."""
    out = {}
    for gid, sub in valid_df.sort_values("date").groupby(group_col, sort=False):
        y_true = sub["sales"].to_numpy()
        if y_true.size == 0:
            continue
        out[gid] = spec(y_true, sub["forecast"].to_numpy(), a1=a1, a2=a2)
    if not out:
        return float("nan")
    if weights is None:
        return float(np.mean(list(out.values())))
    w = {g: weights.get(g, 0.0) for g in out}
    total = sum(w.values())
    if total == 0:
        return float(np.mean(list(out.values())))
    return float(sum(out[g] * w[g] / total for g in out))


def compute_spec(train_df, valid_df, preds, group_col=_SERIES_COL,
                  price_col=_PRICE_COL, a1=0.75, a2=0.25):
    """Adjunta las predicciones a valid_df y calcula el SPEC final (mismo patrón
    que compute_wrmsse)."""
    if group_col not in valid_df.columns:
        return float("nan")

    valid_with_preds = valid_df[[group_col, "date", "sales"]].copy()
    valid_with_preds["forecast"] = clip_closed_stores(valid_df, preds)

    weights = None
    if price_col in train_df.columns:
        weights = compute_weights(train_df, price_col, group_col)

    return calculate_spec(valid_with_preds, weights, group_col, a1=a1, a2=a2)


def compute_mae_scales(train_df, group_col=_SERIES_COL, m=1):
    """Denominador MASE por serie: MAE del naive de m pasos in-sample desde la
    primera venta no-cero (mismo criterio que compute_scales, pero con MAE en
    vez de MSE, siguiendo Hyndman & Koehler, 2006)."""
    scales = {}
    for gid, sub in train_df.sort_values("date").groupby(group_col, sort=False):
        y = sub["sales"].to_numpy(dtype=float)
        nz = np.flatnonzero(y)
        if nz.size == 0:
            continue
        y = y[nz[0]:]
        if y.size <= m:
            continue
        scales[gid] = float(np.mean(np.abs(y[m:] - y[:-m])))
    return scales


def calculate_mase(valid_df, scales, group_col=_SERIES_COL):
    """MASE por serie. valid_df necesita: group_col, sales, forecast."""
    out = {}
    for gid, sub in valid_df.groupby(group_col, sort=False):
        scale = scales.get(gid)
        if not scale:
            continue
        mae_i = np.mean(np.abs(sub["sales"].to_numpy() - sub["forecast"].to_numpy()))
        out[gid] = float(mae_i / scale)
    return out


def calculate_wmase(valid_df, scales, weights=None, group_col=_SERIES_COL):
    """MASE ponderado (mismo patrón que calculate_wrmsse). weights=None → promedio."""
    mase = calculate_mase(valid_df, scales, group_col)
    if not mase:
        return float("nan")
    if weights is None:
        return float(np.mean(list(mase.values())))
    w = {g: weights.get(g, 0.0) for g in mase}
    total = sum(w.values())
    if total == 0:
        return float(np.mean(list(mase.values())))
    return float(sum(mase[g] * w[g] / total for g in mase))


def compute_mase(train_df, valid_df, preds, group_col=_SERIES_COL,
                  price_col=_PRICE_COL, m=1):
    """Adjunta las predicciones a valid_df y calcula el MASE final (mismo patrón
    que compute_wrmsse)."""
    if group_col not in valid_df.columns:
        return float("nan")

    valid_with_preds = valid_df[[group_col, "date", "sales"]].copy()
    valid_with_preds["forecast"] = clip_closed_stores(valid_df, preds)

    scales = compute_mae_scales(train_df, group_col, m=m)
    weights = None
    if price_col in train_df.columns:
        weights = compute_weights(train_df, price_col, group_col)

    return calculate_wmase(valid_with_preds, scales, weights, group_col)


def clip_closed_stores(valid_df, preds):
    """Fuerza el forecast a 0 en filas con is_store_closed=1 (cierre conocido de
    antemano vía calendario, no derivado de las ventas: no hay leakage). Cubre en
    predicción lo que el procesamiento ya hace en el target de entrenamiento."""
    preds = np.asarray(preds, dtype=float)
    if "is_store_closed" in valid_df.columns:
        preds = np.where(valid_df["is_store_closed"].to_numpy() == 1, 0.0, preds)
    return preds


def compute_wrmsse(train_df, valid_df, preds, group_col=_SERIES_COL,
                    price_col=_PRICE_COL):
    """Adjunta las predicciones a valid_df y calcula el WRMSSE final."""
    if group_col not in valid_df.columns:
        return float("nan")

    valid_with_preds = valid_df[[group_col, "date", "sales"]].copy()
    valid_with_preds["forecast"] = clip_closed_stores(valid_df, preds)

    scales = compute_scales(train_df, group_col)
    weights = None
    if price_col in train_df.columns:
        weights = compute_weights(train_df, price_col, group_col)

    return calculate_wrmsse(valid_with_preds, scales, weights, group_col)


def mae(y_true, y_pred):
    """MAE: error absoluto medio."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """RMSE: raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-8):
    """MAPE: error porcentual absoluto medio (con epsilon para evitar división por 0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))


def wape_metric(y_true, y_pred):
    """Eval metric de LightGBM (name, score, is_higher_better) para WAPE."""
    return "wape", wape(y_true, y_pred), False


def make_wrmsse_metric(train_df, valid_df):
    """Fábrica de eval metric de LightGBM para WRMSSE: cierra sobre train/valid ya
    que la callback de lgb sólo recibe (y_true, y_pred)."""
    def _wrmsse_metric(y_true, y_pred):
        return "wrmsse", compute_wrmsse(train_df, valid_df, y_pred), False
    return _wrmsse_metric


def evaluate_predictions(train_df, valid_df, y_valid, y_pred_valid, name, fit_time=None, category=None):
    """WAPE/WRMSSE de un modelo sobre validación (clipando cierres conocidos).
    Pensada para acumular en una lista y comparar modelos, p.ej.:
    `model_results.append(evaluate_predictions(train, valid, y_valid, model.predict(X_valid), "LightGBM"))`

    `fit_time` (segundos, opcional) permite comparar también el costo de
    entrenamiento de cada modelo.
    `category` (opcional) permite etiquetar el modelo (p.ej. "Naive", "ML")
    para filtrar/comparar resultados por familia.
    """
    y_pred_valid = clip_closed_stores(valid_df, y_pred_valid)
    result = {
        "model": name,
        "category": category,
        "wape": float(wape(y_valid, y_pred_valid)),
        "wrmsse": compute_wrmsse(train_df, valid_df, y_pred_valid),
        "mae": mae(y_valid, y_pred_valid),
        "rmse": rmse(y_valid, y_pred_valid),
        "mape": mape(y_valid, y_pred_valid),
        "smape": float(smape(y_valid, y_pred_valid)),
        "bias": float(bias(y_valid, y_pred_valid)),
        "spec": compute_spec(train_df, valid_df, y_pred_valid),
        "mase": compute_mase(train_df, valid_df, y_pred_valid),
        "fit_time": fit_time,
    }
    label = f"{name} [{category}]" if category else name
    msg = (
        f"{label:>25s} | WAPE: {result['wape']:.2%} | WRMSSE: {result['wrmsse']:.4f} "
        f"| MAE: {result['mae']:.3f} | RMSE: {result['rmse']:.3f} "
        f"| MAPE: {result['mape']:.2%} | SMAPE: {result['smape']:.2%} | Bias: {result['bias']:.2%} "
        f"| SPEC: {result['spec']:.3f} | MASE: {result['mase']:.4f}"
    )
    if fit_time is not None:
        msg += f" | fit: {fit_time:.2f}s"
    print(msg)
    return result


def build_predictions_report(train_df, eval_df, y_true, y_pred, target_col="sales",
                              id_cols=(_SERIES_COL, "date"), extra_cols=("gross_sales",)):
    """Clipa cierres, calcula WAPE/WRMSSE globales y arma el detalle de error por fila.

    Pensada para reusarse en cualquier etapa (modelo simple, post-Optuna, modelo
    final): recibe el modelo ya evaluado (y_pred) en vez de reentrenar.
    """
    y_pred = clip_closed_stores(eval_df, y_pred)

    metrics = {
        "wape": float(wape(y_true, y_pred)),
        "wrmsse": compute_wrmsse(train_df, eval_df, y_pred),
    }

    df_pred = eval_df.copy()
    df_pred["y_pred"] = np.round(y_pred, 0).astype(float)
    df_pred["error"] = df_pred[target_col] - df_pred["y_pred"]
    df_pred["abs_error"] = df_pred["error"].abs()
    df_pred["wape"] = df_pred["abs_error"] / df_pred[target_col].abs()
    df_pred["bias"] = -df_pred["error"] / df_pred[target_col].abs()

    cols = list(id_cols) + [target_col] + list(extra_cols) + ["y_pred", "error", "abs_error", "wape", "bias"]
    df_pred = df_pred[cols].sort_values("abs_error")

    return metrics, df_pred


def build_series_metrics(train_df, df_pred, target_col="sales", group_col=_SERIES_COL,
                          weight_level=("date",)):
    """WAPE/bias/WRMSSE por serie a partir del detalle de predicciones (salida de
    `build_predictions_report`, debe traer las columnas `y_pred` y `gross_sales`).

    `weight_level` define cómo se pondera el error dentro del WRMSSE de cada serie:
    (group_col,) -> sin ponderación real (peso constante = total de la serie, igual a RMSSE)
    ("date",) -> pondera cada fecha por su gross_sales (días de más venta pesan más)
    (group_col, "date") -> pondera cada fila por su propio gross_sales
    """
    weight_level = list(weight_level)
    scales = compute_scales(train_df, group_col)
    mae_scales = compute_mae_scales(train_df, group_col)

    def _wrmsse_for_group(g):
        scale = scales.get(g.name)
        if not scale:
            return np.nan
        weights = g.groupby(weight_level)["gross_sales"].transform("sum").to_numpy()
        sq_err = (g[target_col].to_numpy() - g["y_pred"].to_numpy()) ** 2
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
        "sales": by_series[target_col].sum(),
        "gross_sales": gross_sales_by_series,
        "gross_sales_pct": gross_sales_by_series / df_pred["gross_sales"].sum(),
        "wape": by_series.apply(lambda g: wape(g[target_col], g["y_pred"])),
        "bias": by_series.apply(lambda g: bias(g[target_col], g["y_pred"])),
        "wrmsse": by_series.apply(_wrmsse_for_group),
        "mase": by_series.apply(_mase_for_group),
        "spec": by_series.apply(_spec_for_group),
    })

    return df_metrics.sort_values("gross_sales", ascending=False)
