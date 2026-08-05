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


def compute_wrmsse(train_df, valid_df, preds, group_col=_SERIES_COL,
                    price_col=_PRICE_COL):
    """Adjunta las predicciones a valid_df y calcula el WRMSSE final."""
    if group_col not in valid_df.columns:
        return float("nan")

    valid_with_preds = valid_df[[group_col, "date", "sales"]].copy()
    valid_with_preds["forecast"] = preds

    scales = compute_scales(train_df, group_col)
    weights = None
    if price_col in train_df.columns:
        weights = compute_weights(train_df, price_col, group_col)

    return calculate_wrmsse(valid_with_preds, scales, weights, group_col)
