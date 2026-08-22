import gc
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

import config
from src.data.reader import read_parquet_pl


@dataclass
class DateSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    valid_start: pd.Timestamp
    test_start: pd.Timestamp

    @property
    def train_days(self) -> int:
        return (self.valid_start - self.first_date).days

    @property
    def valid_days(self) -> int:
        return (self.test_start - self.valid_start).days

    @property
    def test_days(self) -> int:
        return (self.last_date - self.test_start).days

    def log_summary(self) -> None:
        logger.info(f"Train : {self.first_date:%Y-%m-%d} to {self.valid_start:%Y-%m-%d} ({self.train_days:,} days, {len(self.train):,} rows)")
        logger.info(f"Valid : {self.valid_start:%Y-%m-%d} to {self.test_start:%Y-%m-%d} ({self.valid_days:,} days, {len(self.valid):,} rows)")
        logger.info(f"Test  : {self.test_start:%Y-%m-%d} to {self.last_date:%Y-%m-%d} ({self.test_days:,} days, {len(self.test):,} rows)")


def date_split(df: pd.DataFrame, valid_days: int, test_days: int, date_col: str = "date") -> DateSplit:
    """Split temporal simple train/valid/test por fecha, sobre un dataframe ya con
    features (a diferencia de `prepare_level`, que arma exógenas/agregados para
    MLForecast). Los últimos `test_days` quedan como test, los `valid_days`
    anteriores como validación, y todo lo previo como train."""
    first_date = df[date_col].min()
    last_date = df[date_col].max()
    test_start = last_date - pd.DateOffset(days=test_days)
    valid_start = test_start - pd.DateOffset(days=valid_days)

    train = df[df[date_col] < valid_start]
    valid = df[(df[date_col] >= valid_start) & (df[date_col] < test_start)]
    test = df[df[date_col] >= test_start]

    return DateSplit(
        train=train, valid=valid, test=test,
        first_date=first_date, last_date=last_date,
        valid_start=valid_start, test_start=test_start,
    )


def build_feature_matrices(df: pd.DataFrame, split: DateSplit, features: list[str],
                           categorical_features: list[str], target: str):
    """Arma X/y para train/valid/test y unifica las categorías de las columnas
    categóricas a partir del dataset completo: evita que cada split termine con un
    set de categorías distinto (rompe modelos que las validan, p. ej. XGBoost)."""
    X_train = split.train[features].copy()
    X_valid = split.valid[features].copy()
    X_test = split.test[features].copy()

    for col in categorical_features:
        categories = df[col].astype("category").cat.categories
        for X in (X_train, X_valid, X_test):
            X[col] = pd.Categorical(X[col], categories=categories)

    y_train = split.train[target]
    y_valid = split.valid[target]
    y_test = split.test[target]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def parse_level_file(file: Path):
    """Extrae (level, grain) del nombre del parquet.

    Formato esperado: level_{id}_{grain}_{name}.parquet
    Devuelve None si el archivo no coincide con el formato esperado (p.ej. archivos legacy).
    """
    m_lvl   = re.search(r"level_(\d+)", file.stem)
    m_grain = re.search(r"level_\d+_(daily|weekly)", file.stem)
    if not m_lvl or not m_grain:
        return None
    return config.LEVELS_BY_ID[int(m_lvl.group(1))], m_grain.group(1)


@dataclass
class LevelSplit:
    train_pd:         pd.DataFrame
    future_exog:      pd.DataFrame
    valid_pd:         pd.DataFrame
    test_pd:          pd.DataFrame
    train_for_wrmsse: pd.DataFrame | None  # None para targets acumulados
    static_cols:      list[str]
    avail_exog:       list[str]
    n_series:         int


def prepare_level(file: Path, level, grain: str, horizon: int,
                  target: str = "sales") -> LevelSplit:
    """Carga y divide el dataset en train/valid/test.

    target selecciona la columna a usar como "y" ('sales' o 'cumN'); para targets
    acumulados se descartan primero las filas finales de cada serie donde cumN es NULL
    (ventana forward incompleta), y train_for_wrmsse es None (WRMSSE no aplica).

    horizon: tamaño en períodos (días si daily, semanas si weekly) de CADA uno de los
    dos bloques reservados al final de la serie. valid y test tienen así la misma
    longitud y son comparables entre sí; train es todo lo anterior. valid está pensado
    para tuning (Optuna, early stopping) y test como prueba final, nunca usado en el
    modelo.
    """
    exog_cols = [config.PRICE_LAG_COL[grain]] + config.EXOG_COLS

    df = (
        read_parquet_pl(file)
        .filter(pl.col(target).is_not_null())
        .rename({"agg_id": "unique_id", "date": "ds", target: "y"})
    )
    n_series = df["unique_id"].n_unique()

    static_cols = [d for d in level.dims if d in df.columns and d not in config.EXCLUDE_AS_STATIC]
    avail_exog  = [c for c in exog_cols if c in df.columns]

    block_days   = timedelta(days=horizon * 7 if grain == "weekly" else horizon)
    max_ds       = df["ds"].max()
    train_cutoff = max_ds - 2 * block_days
    test_cutoff  = max_ds - block_days

    train_pl = df.filter(pl.col("ds") <= train_cutoff)
    valid_pl = df.filter((pl.col("ds") > train_cutoff) & (pl.col("ds") <= test_cutoff))
    test_pl  = df.filter(pl.col("ds") > test_cutoff)
    del df; gc.collect()

    # WRMSSE requiere ventas individuales; no aplica para targets acumulados
    if target == "sales":
        wrmsse_cols = ["unique_id", "ds", "y"] + (
            ["avg_sell_price"] if "avg_sell_price" in train_pl.columns else []
        )
        train_for_wrmsse = (
            train_pl.select(wrmsse_cols)
            .rename({"unique_id": "agg_id", "ds": "date", "y": "sales"})
            .to_pandas()
        )
    else:
        train_for_wrmsse = None

    model_cols = ["unique_id", "ds", "y"] + static_cols + avail_exog
    train_pd    = train_pl.select([c for c in model_cols if c in train_pl.columns]).to_pandas()
    future_exog = (
        pl.concat([valid_pl, test_pl])
        .select(["unique_id", "ds"] + avail_exog)
        .to_pandas()
    )
    valid_pd = valid_pl.select(["unique_id", "ds", "y"]).to_pandas()
    test_pd  = test_pl.select(["unique_id", "ds", "y"]).to_pandas()
    del train_pl, valid_pl, test_pl; gc.collect()

    return LevelSplit(
        train_pd=train_pd,
        future_exog=future_exog,
        valid_pd=valid_pd,
        test_pd=test_pd,
        train_for_wrmsse=train_for_wrmsse,
        static_cols=static_cols,
        avail_exog=avail_exog,
        n_series=n_series,
    )
