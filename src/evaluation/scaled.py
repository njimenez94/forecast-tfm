"""Métricas escaladas/jerárquicas por serie (RMSSE/WRMSSE, MASE/WMASE, SPEC),
que necesitan escalas y pesos calculados sobre train."""
import numpy as np
import pandas as pd

_SERIES_COL = "series_id"
_PRICE_COL = "avg_sell_price"


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


def compute_naive_scales(train_df, group_col=_SERIES_COL, target_col="sales", m=1):
    """Denominador RMSSE por serie: MSE del naive de m pasos (m=1 para 'sales'; m=N
    para el target acumulado cumN, comparando cada ventana de N períodos con la
    anterior no solapada) calculado in-sample sobre train, desde la primera venta
    no-cero de la serie (siempre según la columna 'sales', para arrancar en el mismo
    punto sea cual sea target_col)."""
    scales = {}
    for gid, sub in train_df.sort_values("date").groupby(group_col, sort=False):
        sales = sub["sales"].to_numpy(dtype=float)
        nz = np.flatnonzero(sales)
        if nz.size == 0:
            continue
        y = sub[target_col].to_numpy(dtype=float)[nz[0]:]
        if y.size <= m:
            continue
        scales[gid] = float(np.mean((y[m:] - y[:-m]) ** 2))
    return scales


def compute_weights(train_df, price_col=_PRICE_COL, group_col=_SERIES_COL,
                    last_days=28):
    """Peso por serie = ventas en $ de los últimos `last_days` del train."""
    cutoff = train_df["date"].max() - pd.Timedelta(days=last_days)
    recent = train_df[train_df["date"] > cutoff]
    dollar = (recent["sales"] * recent[price_col]).groupby(recent[group_col]).sum()
    return dollar.to_dict()


def calculate_rmsse(valid_df, scales, group_col=_SERIES_COL, target_col="sales"):
    """RMSSE por serie. valid_df necesita: group_col, target_col, forecast."""
    out = {}
    for gid, sub in valid_df.groupby(group_col, sort=False):
        scale = scales.get(gid)
        if not scale:
            continue
        num = np.mean((sub[target_col].to_numpy() - sub["forecast"].to_numpy()) ** 2)
        out[gid] = float(np.sqrt(num / scale))
    return out


def calculate_wrmsse(valid_df, scales, weights=None, group_col=_SERIES_COL, target_col="sales"):
    """WRMSSE = Σ w_i · RMSSE_i (pesos normalizados). weights=None → promedio."""
    rmsse = calculate_rmsse(valid_df, scales, group_col, target_col)
    if not rmsse:
        return float("nan")
    if weights is None:
        return float(np.mean(list(rmsse.values())))
    w = {g: weights.get(g, 0.0) for g in rmsse}
    total = sum(w.values())
    if total == 0:
        return float(np.mean(list(rmsse.values())))
    return float(sum(rmsse[g] * w[g] / total for g in rmsse))


def calculate_spec(valid_df, weights=None, group_col=_SERIES_COL, a1=0.75, a2=0.25, target_col="sales"):
    """SPEC por serie (necesita orden temporal), agregado como promedio ponderado
    igual que WRMSSE. valid_df necesita: group_col, date, target_col, forecast."""
    out = {}
    for gid, sub in valid_df.sort_values("date").groupby(group_col, sort=False):
        y_true = sub[target_col].to_numpy()
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
                  price_col=_PRICE_COL, a1=0.75, a2=0.25, target_col="sales"):
    """Adjunta las predicciones a valid_df y calcula el SPEC final (mismo patrón
    que compute_wrmsse)."""
    if group_col not in valid_df.columns:
        return float("nan")

    valid_with_preds = valid_df[[group_col, "date", target_col]].copy()
    valid_with_preds["forecast"] = clip_closed_stores(valid_df, preds)

    weights = None
    if price_col in train_df.columns:
        weights = compute_weights(train_df, price_col, group_col)

    return calculate_spec(valid_with_preds, weights, group_col, a1=a1, a2=a2, target_col=target_col)


def compute_mae_scales(train_df, group_col=_SERIES_COL, m=1, target_col="sales"):
    """Denominador MASE por serie: MAE del naive de m pasos in-sample desde la
    primera venta no-cero (mismo criterio que compute_naive_scales, pero con MAE en
    vez de MSE, siguiendo Hyndman & Koehler, 2006)."""
    scales = {}
    for gid, sub in train_df.sort_values("date").groupby(group_col, sort=False):
        sales = sub["sales"].to_numpy(dtype=float)
        nz = np.flatnonzero(sales)
        if nz.size == 0:
            continue
        y = sub[target_col].to_numpy(dtype=float)[nz[0]:]
        if y.size <= m:
            continue
        scales[gid] = float(np.mean(np.abs(y[m:] - y[:-m])))
    return scales


def calculate_mase(valid_df, scales, group_col=_SERIES_COL, target_col="sales"):
    """MASE por serie. valid_df necesita: group_col, target_col, forecast."""
    out = {}
    for gid, sub in valid_df.groupby(group_col, sort=False):
        scale = scales.get(gid)
        if not scale:
            continue
        mae_i = np.mean(np.abs(sub[target_col].to_numpy() - sub["forecast"].to_numpy()))
        out[gid] = float(mae_i / scale)
    return out


def calculate_wmase(valid_df, scales, weights=None, group_col=_SERIES_COL, target_col="sales"):
    """MASE ponderado (mismo patrón que calculate_wrmsse). weights=None → promedio."""
    mase = calculate_mase(valid_df, scales, group_col, target_col)
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
                  price_col=_PRICE_COL, m=1, target_col="sales"):
    """Adjunta las predicciones a valid_df y calcula el MASE final (mismo patrón
    que compute_wrmsse)."""
    if group_col not in valid_df.columns:
        return float("nan")

    valid_with_preds = valid_df[[group_col, "date", target_col]].copy()
    valid_with_preds["forecast"] = clip_closed_stores(valid_df, preds)

    scales = compute_mae_scales(train_df, group_col, m=m, target_col=target_col)
    weights = None
    if price_col in train_df.columns:
        weights = compute_weights(train_df, price_col, group_col)

    return calculate_wmase(valid_with_preds, scales, weights, group_col, target_col=target_col)


def clip_closed_stores(valid_df, preds):
    """Fuerza el forecast a 0 en filas con is_store_closed=1 (cierre conocido de
    antemano vía calendario, no derivado de las ventas: no hay leakage). Cubre en
    predicción lo que el procesamiento ya hace en el target de entrenamiento."""
    preds = np.asarray(preds, dtype=float)
    if "is_store_closed" in valid_df.columns:
        preds = np.where(valid_df["is_store_closed"].to_numpy() == 1, 0.0, preds)
    return preds


def compute_wrmsse(train_df, valid_df, preds, group_col=_SERIES_COL,
                    price_col=_PRICE_COL, scales=None, weights=None, target_col="sales", m=1):
    """Adjunta las predicciones a valid_df y calcula el WRMSSE final.

    `scales`/`weights` opcionales: si ya se calcularon antes para el mismo
    `train_df` (p.ej. en `make_wrmsse_metric`, invariantes entre llamadas
    porque no dependen de las predicciones), se reusan en vez de recalcular
    el groupby completo sobre train en cada invocación."""
    if group_col not in valid_df.columns:
        return float("nan")

    valid_with_preds = valid_df[[group_col, "date", target_col]].copy()
    valid_with_preds["forecast"] = clip_closed_stores(valid_df, preds)

    if scales is None:
        scales = compute_naive_scales(train_df, group_col, target_col=target_col, m=m)
    if weights is None and price_col in train_df.columns:
        weights = compute_weights(train_df, price_col, group_col)

    return calculate_wrmsse(valid_with_preds, scales, weights, group_col, target_col=target_col)
