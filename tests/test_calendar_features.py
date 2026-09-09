"""add_calendar_features: quarter/is_month_start/is_month_end (calendario exacto) +
codificación cíclica sin/cos de dow/month/doy (valores conocidos en los bordes del
período, donde el seno/coseno toma valores redondos)."""
import polars as pl

from src.features.calendar import add_calendar_features


def test_calendar_flags_and_cyclical_edges():
    # 2024-01-01 es lunes (dow=1 en la convención del módulo, ver docstring),
    # primer día del año/mes/trimestre -> ángulo 0 en las tres codificaciones cíclicas.
    dates = [
        "2024-01-01",  # lunes, día 1 de mes/trimestre/año -> sin=0, cos=1 en las 3
        "2024-01-31",  # último día de enero (is_month_end)
        "2024-04-01",  # primer día de Q2
        "2024-01-04",  # jueves (dow=4)
    ]
    df = pl.DataFrame({"date": [pl.Series([d]).str.to_date()[0] for d in dates]})
    out = add_calendar_features(df)

    row0 = out.row(0, named=True)
    assert row0["quarter"] == 1
    assert row0["is_month_start"] == 1
    assert row0["is_month_end"] == 0
    assert abs(row0["dow_sin"]) < 1e-6 and abs(row0["dow_cos"] - 1) < 1e-6
    assert abs(row0["month_sin"]) < 1e-6 and abs(row0["month_cos"] - 1) < 1e-6
    assert abs(row0["doy_sin"]) < 1e-6 and abs(row0["doy_cos"] - 1) < 1e-6

    row1 = out.row(1, named=True)
    assert row1["is_month_end"] == 1
    assert row1["is_month_start"] == 0

    row2 = out.row(2, named=True)
    assert row2["quarter"] == 2
    # abril = mes 4 -> angle = 2*pi*3/12 = pi/2 (la codificación es por mes del año,
    # no reinicia en cada trimestre)
    assert abs(row2["month_sin"] - 1) < 1e-6 and abs(row2["month_cos"]) < 1e-6

    row3 = out.row(3, named=True)
    # jueves: dow=4 (1=lunes) -> angle = 2*pi*3/7, ni sin ni cos son 0/1
    assert row3["dow_sin"] != 0 and abs(row3["dow_cos"] - 1) > 1e-6
