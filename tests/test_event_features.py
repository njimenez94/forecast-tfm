"""add_event_features sobre una serie sintética de 6 días con un evento genérico
(Thanksgiving, día 1) y un cierre de tienda (Christmas, día 3): cubre distancia a
evento genérico, distancia con signo a cierre (con la ventana de +-7 días),
zero-out de sales/gross_sales en el cierre, y codificación de event_type."""
import datetime as dt

import polars as pl

from src.features.events import add_event_features


def _dates(n: int) -> list[dt.date]:
    return [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def _base_df() -> pl.DataFrame:
    n = 6
    event_name_1 = [None, "Thanksgiving", None, "Christmas", None, None]
    event_type_1 = [None, "National", None, "National", None, None]
    return pl.DataFrame({
        "series_id": ["S1"] * n,
        "date": _dates(n),
        "avg_sell_price": [10.0] * n,  # ya "liberada" desde el día 0 (sin release a mitad de camino)
        "has_event": [1 if e is not None else 0 for e in event_name_1],
        "has_event_2": [0] * n,
        "event_name_1": event_name_1,
        "event_type_1": event_type_1,
        # dtype explícito: una columna de puros None sin tipo queda como Null y
        # replace_strict (event_type_*_enc) no puede mapearla contra las claves str.
        "event_name_2": pl.Series([None] * n, dtype=pl.Utf8),
        "event_type_2": pl.Series([None] * n, dtype=pl.Utf8),
        "sales": [10.0, 6.0, 10.0, 999.0, 10.0, 10.0],
        "gross_sales": [100.0, 60.0, 100.0, 9999.0, 100.0, 100.0],
    })


def test_store_closure_zeroes_out_sales_only_on_closure_day():
    out = add_event_features(_base_df()).sort("date")
    assert out["is_store_closed"].to_list() == [0, 0, 0, 1, 0, 0]
    assert out["sales"].to_list() == [10.0, 6.0, 10.0, 0.0, 10.0, 10.0]
    assert out["gross_sales"].to_list() == [100.0, 60.0, 100.0, 0.0, 100.0, 100.0]


def test_generic_event_distance_forward_and_backward():
    out = add_event_features(_base_df()).sort("date")
    # eventos en día 1 (Thanksgiving) y día 3 (Christmas): days_since_event cuenta
    # desde el evento más reciente (incluyendo el propio día del evento -> 0),
    # days_to_event hacia el próximo (null después del último evento de la serie).
    assert out["days_since_event"].to_list() == [None, 0, 1, 0, 1, 2]
    assert out["days_to_event"].to_list() == [1, 0, 1, 0, None, None]


def test_signed_days_to_closure_within_window():
    out = add_event_features(_base_df()).sort("date")
    assert out["days_to_closure"].to_list() == [3, 2, 1, 0, -1, -2]


def test_event_type_encoding_and_high_impact_distance():
    out = add_event_features(_base_df()).sort("date")
    assert out["event_type_1_enc"].to_list() == [0, 3, 0, 3, 0, 0]  # National -> 3, sin evento -> 0
    assert out["event_type_2_enc"].to_list() == [0] * 6  # sin event_type_2 en la serie
    # Thanksgiving (día 1): días_since/to solo para ese evento puntual, no genérico
    assert out["days_since_thanksgiving"].to_list() == [None, 0, 1, 2, 3, 4]
    assert out["days_to_thanksgiving"].to_list() == [1, 0, None, None, None, None]


def test_days_since_release_is_zero_when_already_released_from_day_zero():
    out = add_event_features(_base_df()).sort("date")
    assert out["days_since_release"].to_list() == [0, 1, 2, 3, 4, 5]
