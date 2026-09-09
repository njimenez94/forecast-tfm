"""add_lag_features: valida el contrato con mlforecast (nombres de columna lag{N},
NaN para historia insuficiente en vez de recorte de filas -- dropna=False, ver
docstring del módulo) sobre una sola serie con valores conocidos, sin intentar
reproducir a mano todas las transforms de config.MLFORECAST_LAG_TRANSFORMS."""
import pandas as pd

from src.features.lags import add_lag_features


def _synthetic_series(n: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "series_id": ["S1"] * n,
        "date": dates,
        "sales": list(range(n)),  # sales[i] == i, para poder predecir lag_k a mano
    })


def test_lag_columns_shift_sales_by_the_right_number_of_periods():
    df = _synthetic_series()
    out = add_lag_features(df, grain="daily", static_cols=[])

    assert "lag1" in out.columns and "lag7" in out.columns
    lag1 = out.sort_values("date")["lag1"].to_numpy()
    lag7 = out.sort_values("date")["lag7"].to_numpy()

    assert pd.isna(lag1[0])
    assert (lag1[1:] == list(range(0, 39))).all()  # lag1[i] == sales[i-1] == i-1

    assert pd.isna(lag7[:7]).all()
    assert (lag7[7:] == list(range(0, 33))).all()  # lag7[i] == sales[i-7] == i-7


def test_add_lag_features_keeps_every_row_even_without_history():
    # dropna=False (ver docstring): filas con historia insuficiente quedan con NaN
    # en vez de descartarse -- lo consume otra vez scripts/train_dataset.py sobre
    # el mismo parquet, así que no se puede perder historia acá.
    df = _synthetic_series(n=10)
    out = add_lag_features(df, grain="daily", static_cols=[])
    assert len(out) == len(df)
