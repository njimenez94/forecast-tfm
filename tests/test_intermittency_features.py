"""add_intermittency_features sobre una serie sintética de 10 días con una racha
de 7 ceros seguida de una venta: cubre zero_streak, adi_expanding y el gating
causal de is_likely_stockout (que debe activarse el día de la venta que rompe la
racha, no el día siguiente, porque se basa en sales.shift(1) -- ver docstring del
módulo sobre por qué todo mira "hasta ayer")."""
import datetime as dt

import polars as pl

from src.features.intermittency import add_intermittency_features


def _dates(n: int) -> list[dt.date]:
    return [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def _base_df() -> pl.DataFrame:
    # release en el día 0 (days_since_release = 0..9); racha de 7 ceros (días 1-7)
    # entre dos días de venta positiva (días 0 y 8); día 9 vuelve a cero.
    sales = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]
    n = len(sales)
    return pl.DataFrame({
        "series_id": ["S1"] * n,
        "date": _dates(n),
        "sales": sales,
        "avg_sell_price": [10.0] * n,
        "is_store_closed": [0] * n,
        "days_since_release": list(range(n)),
    })


def test_zero_streak_counts_consecutive_zero_days_before_today():
    out = add_intermittency_features(_base_df()).sort("date")
    # día 0: sin historia previa -> null. Después, cuenta días de racha *hasta ayer*
    # (no incluye hoy): sube de 0 a 6 durante la racha, y refleja la racha completa
    # (7) el día que la venta la rompe, porque no mira la venta de hoy.
    assert out["zero_streak"].to_list() == [None, 0, 1, 2, 3, 4, 5, 6, 7, 0]


def test_adi_expanding_drops_after_a_new_positive_sale():
    out = add_intermittency_features(_base_df()).sort("date")
    adi = out["adi_expanding"].to_list()
    assert adi[0] is None  # sin ventas previas todavía
    assert adi[2] == 1.0   # (días_elapsed=1)/(1 venta previa)
    assert adi[7] == 6.0   # racha larga sin nueva venta -> ADI crece
    assert adi[9] == 4.0   # la venta del día 8 duplica el denominador -> ADI baja


def test_is_likely_stockout_flags_the_day_the_streak_breaks_not_the_day_after():
    out = add_intermittency_features(_base_df()).sort("date")
    # umbral = 7 días de racha (_STOCKOUT_STREAK_THRESHOLD); solo el día 8 acumula
    # zero_streak >= 7 -- pese a tener sales=10 ese mismo día, porque el flag no mira
    # la venta de hoy (ver docstring del módulo).
    assert out["is_likely_stockout"].to_list() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]


def test_is_likely_stockout_respects_store_closure_and_missing_price():
    df = _base_df()
    # si la tienda está cerrada o el precio es null ese día, no debe marcarse como
    # quiebre de stock aunque la racha de ceros llegue al umbral.
    df_closed = df.with_columns(
        pl.when(pl.col("date") == _dates(10)[8]).then(1).otherwise(pl.col("is_store_closed"))
        .alias("is_store_closed"),
    )
    out_closed = add_intermittency_features(df_closed).sort("date")
    assert out_closed["is_likely_stockout"].to_list()[8] == 0
