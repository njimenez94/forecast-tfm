"""Baselines ingenuos: no entrenan nada, son el piso mínimo que cualquier
modelo debe superar (ver notebook 02_model, sección "Comparación de modelos")."""
import numpy as np
import pandas as pd


def seasonal_naive(train_df, valid_df, target_col, group_col="series_id", season_length=7):
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


def drift(train_df, valid_df, target_col, group_col="series_id"):
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


def historical_mean(train_df, valid_df, target_col, group_col="series_id"):
    """Promedio histórico completo de train por serie (constante en todo el
    horizonte de validación); solo tiene sentido si la serie es ~estacionaria."""
    hist_mean = train_df.groupby(group_col)[target_col].mean()
    return valid_df[group_col].map(hist_mean).fillna(0.0).to_numpy()


def moving_average(train_df, valid_df, target_col, group_col="series_id", window=7):
    """Promedio de los últimos `window` valores observados en train por serie
    (suaviza ruido reciente), repetido de forma constante en validación."""
    ma = (
        train_df.sort_values("date")
        .groupby(group_col)[target_col]
        .apply(lambda s: s.to_numpy(dtype=float)[-window:].mean())
    )
    return valid_df[group_col].map(ma).fillna(0.0).to_numpy()
