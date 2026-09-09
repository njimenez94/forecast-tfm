"""add_price_features: price_vs_max (nivel relativo al máximo histórico de la
serie), price_volatility (std móvil) y days_since_price_change (forward-fill del
último cambio) -- valores de referencia calculados con numpy/a mano, no copiados
de la implementación, para no validar el código contra sí mismo."""
import datetime as dt

import numpy as np
import polars as pl

from src.features.price import add_price_features


def _dates(n: int) -> list[dt.date]:
    return [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def test_price_vs_max_is_relative_to_running_max():
    prices = [10.0, 20.0, 15.0, 20.0, 5.0]
    df = pl.DataFrame({
        "series_id": ["S1"] * len(prices),
        "date": _dates(len(prices)),
        "avg_sell_price": prices,
    })
    out = add_price_features(df).sort("date")

    running_max = np.maximum.accumulate(prices)
    expected = np.array(prices) / running_max
    assert np.allclose(out["price_vs_max"].to_numpy(), expected, atol=1e-5)
    # el propio día del máximo histórico siempre vale 1.0 (precio == máximo hasta ese día)
    assert out["price_vs_max"][1] == 1.0


def test_price_volatility_matches_numpy_rolling_std():
    prices = [10.0, 20.0, 10.0, 20.0, 30.0]
    df = pl.DataFrame({
        "series_id": ["S1"] * len(prices),
        "date": _dates(len(prices)),
        "avg_sell_price": prices,
    })
    out = add_price_features(df).sort("date")
    vol = out["price_volatility"].to_numpy()

    assert np.isnan(vol[0])  # min_samples=2: el primer día no tiene ventana suficiente
    for i in range(1, len(prices)):
        expected = np.std(prices[: i + 1], ddof=1)
        assert abs(vol[i] - expected) < 1e-4


def test_days_since_price_change_resets_on_each_change():
    # precio estable 2 días, cambia el día 2, estable el resto -> días desde el
    # cambio: [null, null, 0, 1, 2]
    prices = [10.0, 10.0, 20.0, 20.0, 20.0]
    df = pl.DataFrame({
        "series_id": ["S1"] * len(prices),
        "date": _dates(len(prices)),
        "avg_sell_price": prices,
    })
    out = add_price_features(df).sort("date")
    days_since = out["days_since_price_change"].to_list()

    assert days_since[0] is None and days_since[1] is None
    assert days_since[2:] == [0, 1, 2]


def test_price_features_are_independent_per_series():
    # dos series con historiales de precio distintos no deben mezclarse (.over("series_id"))
    df = pl.DataFrame({
        "series_id": ["S1", "S1", "S2", "S2"],
        "date": _dates(2) * 2,
        "avg_sell_price": [10.0, 5.0, 100.0, 100.0],
    })
    out = add_price_features(df).sort(["series_id", "date"])
    s1 = out.filter(pl.col("series_id") == "S1")["price_vs_max"].to_list()
    s2 = out.filter(pl.col("series_id") == "S2")["price_vs_max"].to_list()
    assert s1 == [1.0, 0.5]
    assert s2 == [1.0, 1.0]
