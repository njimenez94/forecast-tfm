"""Relocalizado desde el antiguo `if __name__ == "__main__":` de
src/features/encoding.py (mismo criterio que tests/test_temporal_split.py)."""
import datetime as _dt

import polars as pl

from src.features.encoding import add_encoding_features


def test_target_encoding_shrinkage_and_conditional_means():
    # I1 (serie S1) vende consistente; I2 (serie S2) nunca vende -- mismo dept/cat
    # (2 series), dept/cat quedan en (I1+I2)/2 -- misma unidad que un item individual,
    # por eso mean() y no sum() en _rolling_target_encoding (ver docstring del módulo).
    dates = [_dt.date(2020, 1, 1) + _dt.timedelta(days=i) for i in range(5)]
    df = pl.DataFrame({
        "series_id": ["S1"] * 5 + ["S2"] * 5,
        "item_id": ["I1"] * 5 + ["I2"] * 5,
        "dept_id": ["D1"] * 10,
        "cat_id": ["C1"] * 10,
        "date": dates * 2,
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "is_weekend": [0, 0, 0, 0, 1] * 2,
        "snap": [0] * 10,
        "has_event": [0] * 10,
    })
    out = add_encoding_features(df)

    # día 1: sin historia previa del item -> null (no leakea el propio día)
    assert out.filter(pl.col("date") == dates[0])["te_item_mean"].is_null().all()

    # dept/cat (raw, antes del shrink) = mean(I1, I2) por día = [5, 10, 15, 20, 25]
    # (shifted). I1 (child=10,15,20,25; n=1,2,3,4) shrinkeado hacia ese dept:
    # (n*child + M*parent)/(n+M), M=_SHRINK_PRIOR_WEIGHT=10.
    i1 = out.filter(pl.col("item_id") == "I1").sort("date")
    assert abs(i1["te_item_mean"][2] - 8.75) < 1e-2      # n=2: (2*15+10*7.5)/12
    assert abs(i1["te_item_mean"][4] - 225 / 14) < 1e-2  # n=4: (4*25+10*12.5)/14

    # I2 (child=0 siempre) tirado hacia el mismo dept:
    i2 = out.filter(pl.col("item_id") == "I2").sort("date")
    assert abs(i2["te_item_mean"][1] - 50 / 11) < 1e-2    # n=1: (0+10*5)/11
    assert abs(i2["te_item_mean"][2] - 6.25) < 1e-2       # n=2: (0+10*7.5)/12
    assert abs(i2["te_item_mean"][4] - 125 / 14) < 1e-2   # n=4: (0+10*12.5)/14

    # S1, weekend=0: media expandiendo de sales previas con el mismo flag (10, 20, 30)
    s1 = out.filter(pl.col("series_id") == "S1").sort("date")
    assert s1["te_weekend_mean"][0] is None
    assert s1["te_weekend_mean"][3] == 20.0  # mean(10, 20, 30)
    assert s1["te_weekend_mean"][4] is None  # primer día con weekend=1 en S1
