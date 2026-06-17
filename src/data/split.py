import gc
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl

import config
from src.data.reader import read_parquet_pl


def parse_level_file(file: Path):
    """Extrae (level, grain, wlabel, target) del nombre del parquet.

    Formato esperado: dataset_level_{id}_{grain}_{window}_{target}_{name}.parquet
    target es 'sales' o 'cumN' (e.g. 'cum7', 'cum365').
    Devuelve None si el archivo no coincide con el formato esperado (p.ej. archivos legacy).
    """
    m_lvl    = re.search(r"level_(\d+)", file.stem)
    m_grain  = re.search(r"level_\d+_(daily|weekly)", file.stem)
    m_window = re.search(r"(?:daily|weekly)_(w[^_]+)_", file.stem)
    m_target = re.search(r"(?:daily|weekly)_w[^_]+_(sales|cum\d+)_", file.stem)
    if not m_lvl or not m_grain or not m_window or not m_target:
        return None
    return (
        config.LEVELS_BY_ID[int(m_lvl.group(1))],
        m_grain.group(1),
        m_window.group(1),
        m_target.group(1),
    )


@dataclass
class LevelSplit:
    train_pd:         pd.DataFrame
    future_exog:      pd.DataFrame
    valid_pd:         pd.DataFrame
    train_for_wrmsse: pd.DataFrame | None  # None para targets acumulados
    static_cols:      list[str]
    avail_exog:       list[str]
    n_series:         int


def prepare_level(file: Path, level, grain: str, horizon: int,
                  target: str = "sales") -> LevelSplit:
    """Carga y divide el dataset en train/valid.

    Para targets acumulados ('cumN'), train_for_wrmsse es None (WRMSSE no aplica).
    horizon: períodos a reservar para validación (días si daily, semanas si weekly).
    """
    exog_cols = [config.PRICE_LAG_COL[grain]] + config.EXOG_COLS

    df       = read_parquet_pl(file).rename({"agg_id": "unique_id", "date": "ds", "sales": "y"})
    n_series = df["unique_id"].n_unique()

    static_cols = [d for d in level.dims if d in df.columns and d not in config.EXCLUDE_AS_STATIC]
    avail_exog  = [c for c in exog_cols if c in df.columns]

    delta_days = horizon * 7 if grain == "weekly" else horizon
    cutoff   = df["ds"].max() - timedelta(days=delta_days)
    train_pl = df.filter(pl.col("ds") <= cutoff)
    valid_pl = df.filter(pl.col("ds") > cutoff)
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
    future_exog = valid_pl.select(["unique_id", "ds"] + avail_exog).to_pandas()
    valid_pd    = valid_pl.select(["unique_id", "ds", "y"]).to_pandas()
    del train_pl, valid_pl; gc.collect()

    return LevelSplit(
        train_pd=train_pd,
        future_exog=future_exog,
        valid_pd=valid_pd,
        train_for_wrmsse=train_for_wrmsse,
        static_cols=static_cols,
        avail_exog=avail_exog,
        n_series=n_series,
    )
