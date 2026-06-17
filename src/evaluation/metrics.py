import numpy as np
import pandas as pd

_SERIES_COL = "agg_id"
_PRICE_COL = "avg_sell_price"


def wape(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()


def bias(y, p):
    return (p - y).sum() / np.abs(y).sum()


def smape(y, p):
    return np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + 1e-8))


# ── M5 WRMSSE ────────────────────────────────────────────────────────────────

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


def compute_wrmsse(train_df, valid_df, preds,
                   group_col=_SERIES_COL, price_col=_PRICE_COL):
    """Wrapper: adjunta forecast a valid_df y calcula WRMSSE."""
    if group_col not in valid_df.columns:
        return float("nan")
    vdf = valid_df[[group_col, "date", "sales"]].copy()
    vdf["forecast"] = preds
    scales = compute_scales(train_df, group_col)
    weights = (compute_weights(train_df, price_col, group_col)
               if price_col in train_df.columns else None)
    return calculate_wrmsse(vdf, scales, weights, group_col)
