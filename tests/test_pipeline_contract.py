"""Tests de contrato sobre src.features.pipeline.build_dataset(): el schema que
consumen scripts/train_dataset.py y api/ (config.EXOG_COLS) tiene que seguir
apareciendo completo en la salida, el pipeline no debe perder/duplicar filas, y
tiene que ser determinista (correrlo dos veces sobre el mismo input da el mismo
resultado -- la noción de "paridad train/serving" que aplica acá, ver plan: la API
no recalcula features online, consume el mismo build_dataset() que el training).

El input sintético reproduce el schema real de data/processed/ (ver
src/data/base_query.py::build_exog_query), no un subconjunto arbitrario -- así el
test detecta si build_dataset() empieza a asumir una columna que el SQL no
garantiza, o viceversa.
"""
import polars as pl

import config
from src.features.pipeline import build_dataset


def _base_processed_df(n_days: int = 40) -> pl.DataFrame:
    dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 1).dt.offset_by(f"{n_days - 1}d"),
                          interval="1d", eager=True)
    series = [
        {"series_id": "CA_1", "store_id": "CA_1", "state_id": "CA", "price": 10.0},
        {"series_id": "CA_2", "store_id": "CA_2", "state_id": "CA", "price": 20.0},
    ]
    frames = []
    for s in series:
        n = len(dates)
        frames.append(pl.DataFrame({
            "series_id": [s["series_id"]] * n,
            "item_id": ["TOTAL"] * n,
            "dept_id": ["TOTAL"] * n,
            "cat_id": ["TOTAL"] * n,
            "store_id": [s["store_id"]] * n,
            "state_id": [s["state_id"]] * n,
            "date": dates,
            "sales": [float(i % 5) for i in range(n)],
            "gross_sales": [float(i % 5) * s["price"] for i in range(n)],
            "avg_sell_price": [s["price"]] * n,
            "price_lag_7": [s["price"]] * n,
            "price_change": [0.0] * n,
            "price_vs_mean": [1.0] * n,
            "has_event": [0] * n,
            "has_event_2": [0] * n,
            "snap": [0] * n,
            "event_name_1": pl.Series([None] * n, dtype=pl.Utf8),
            "event_type_1": pl.Series([None] * n, dtype=pl.Utf8),
            "event_name_2": pl.Series([None] * n, dtype=pl.Utf8),
            "event_type_2": pl.Series([None] * n, dtype=pl.Utf8),
        }))
    df = pl.concat(frames)
    return df.with_columns(
        pl.col("date").dt.year().cast(pl.Int16).alias("year"),
        pl.col("date").dt.month().cast(pl.Int8).alias("month"),
        pl.col("date").dt.day().cast(pl.Int8).alias("day"),
        pl.col("date").dt.weekday().cast(pl.Int8).alias("dayofweek"),
        pl.col("date").dt.week().cast(pl.Int8).alias("weekofyear"),
        (pl.col("date").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
    )


def test_exog_cols_are_all_present_in_the_built_dataset():
    static_cols = ["store_id", "state_id"]
    out = build_dataset(_base_processed_df(), "daily", static_cols)
    missing = [c for c in config.EXOG_COLS if c not in out.columns]
    assert not missing, f"config.EXOG_COLS con columnas que build_dataset() ya no produce: {missing}"


def test_build_dataset_does_not_drop_or_duplicate_rows():
    df = _base_processed_df()
    out = build_dataset(df, "daily", ["store_id", "state_id"])
    assert len(out) == len(df)
    assert not out.duplicated(subset=["series_id", "date"]).any()


def test_build_dataset_downcasts_floats_and_strings():
    out = build_dataset(_base_processed_df(), "daily", ["store_id", "state_id"])
    assert (out.select_dtypes("float64").shape[1]) == 0
    assert out["avg_sell_price"].dtype.name == "float32"
    assert out["item_id"].dtype.name == "category"


def test_build_dataset_is_deterministic_across_reruns():
    # mismo input -> mismo output, bit a bit: es la garantía real de "paridad
    # train/serving" en este proyecto (api/ no recalcula features, ver
    # api/main.py -- lo que hay que blindar es que este pipeline no dependa de
    # nada no determinista, p. ej. orden de filas o de columnas).
    df = _base_processed_df()
    out1 = build_dataset(df, "daily", ["store_id", "state_id"])
    out2 = build_dataset(df, "daily", ["store_id", "state_id"])
    assert list(out1.columns) == list(out2.columns)
    assert out1.equals(out2)
