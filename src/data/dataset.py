import gc
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from loguru import logger

import config
from src.data.temporal_split import build_feature_matrices, date_split
from src.evaluation import make_wrmsse_metric


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


def load_data(level_id: int, target: str, dataset_path: Path | None = None,
               grain: str | None = None) -> SimpleNamespace:
    """Carga el dataset final (artifacts/datasets/) de un nivel/target y arma las
    listas de features (categóricas primero) que usan build_feature_matrices/
    scripts.train_dataset. `dataset_path`: override para niveles con split_by
    (10-12), donde no hay un único parquet por nivel -- ver
    scripts/train_dataset.py main()/config.featured_level_combos(). `grain`: cuál
    de `level.grains` cargar (default: el primero) -- lo decide el caller
    (Config.grain en train_dataset.py); antes se asumía siempre level.grains[0]
    acá, lo que hacía que un `target` "weekly" terminara entrenando igual sobre
    el dataset daily."""
    level = config.LEVELS_BY_ID[level_id]
    grain = grain or level.grains[0]
    path = dataset_path or config.featured_level_path(level, grain)
    level_str = path.stem.removeprefix("dataset_")

    df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)

    id_cols = ["series_id", "date"]
    leaky_cols = ["gross_sales"]

    features = [c for c in df.columns if c not in id_cols + leaky_cols + [target]]
    categorical_features = [
        c for c in [
            "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        ] if c in df.columns
    ]
    numerical_features = [c for c in features if c not in categorical_features]
    features = categorical_features + numerical_features

    logger.info("{} | target={} | {:,} filas | {} features ({} categóricas)",
                level_str, target, len(df), len(features), len(categorical_features))

    return SimpleNamespace(
        level=level, grain=grain, level_str=level_str, df=df, target=target,
        # m: paso del naive scale en WRMSSE/MASE -- ver src.evaluation.scaled.compute_naive_scales.
        m=1,
        id_cols=id_cols, leaky_cols=leaky_cols,
        features=features, categorical_features=categorical_features,
        numerical_features=numerical_features,
        model_results=[], predictions_valid={},
    )


def split_data(state: SimpleNamespace, target: str, test_start: pd.Timestamp | None = None) -> None:
    """Parte state.df (de load_data) en train/valid/test + X/y de cada split, y arma
    state.wrmsse_metric (eval metric de LightGBM). Muta `state` in place."""
    # Trunca a config.SPLIT_DATES["test"]["end"] en vez de dejar que date_split derive
    # last_date de df["date"].max(): en weekly, la fila de la última semana calendario
    # puede tener menos de 7 días de venta real (ver config.SPLIT_DATES) -- sin este
    # corte, esa semana parcial se colaría en test.
    state.df = state.df[state.df["date"] <= config.SPLIT_DATES["test"][state.grain]["end"]]
    if test_start is None:
        # No aplica si el caller ya pasó un test_start explícito (reconstruct_test_data()
        # reproduce un split ya resuelto -- guardado tal cual en el artifact, no hay que
        # volver a correrlo).
        test_start = config.SPLIT_DATES["test"][state.grain]["start"]
    split = date_split(
        state.df,
        valid_days=config.valid_year_days(state.grain),
        test_days=config.test_days(state.grain),
        test_start=test_start,
    )
    split.log_summary()

    X_train, y_train, X_valid, y_valid, X_test, y_test = build_feature_matrices(
        state.df, split, state.features, state.categorical_features, state.target, state.grain,
    )
    state.X_train, state.y_train = X_train, y_train
    state.X_valid, state.y_valid = X_valid, y_valid
    state.X_test, state.y_test = X_test, y_test

    # train/valid/test solo se usan después para bookkeeping de evaluación (scales
    # WRMSSE/MASE, pesos por precio, clip de cierres, gross_sales de reportes) --
    # no las ~100 columnas de features (esas ya están en X_train/X_valid/X_test).
    # Cargar el nivel 12 completo (mayor cardinalidad) puede acercarse al límite de
    # RAM disponible; recortar acá evita cargar ese peso tres veces más.
    eval_cols = [c for c in dict.fromkeys(
        ["series_id", "date", "sales", "gross_sales", "avg_sell_price", "is_store_closed", state.target]
    ) if c in state.df.columns]
    state.train = split.train[eval_cols].copy()
    state.valid = split.valid[eval_cols].copy()
    state.test = split.test[eval_cols].copy()
    state.first_date, state.valid_start, state.test_start = (
        split.first_date, split.valid_start, split.test_start,
    )
    del split, state.df
    gc.collect()

    state.wrmsse_metric = make_wrmsse_metric(state.train, state.valid, target_col=state.target, m=state.m)

    logger.info("Features: {} ({} categóricas)", len(state.features), len(state.categorical_features))


def reconstruct_test_data(artifact: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Reconstruye X_test/y_test/train/test de un artifact liviano (export_artifact ya
    no las guarda), repitiendo load_data()+split_data() sobre el parquet de
    data/datasets/ (make build-datasets). Usado por scripts/build_test_sets.py y
    notebooks/03_predictions.ipynb.

    Fija el split al `test_start` guardado en el artifact (en vez de derivarlo del
    último día del parquet, el default de date_split): reproduce el split exacto
    usado en el entrenamiento aunque el parquet se haya regenerado después.

    `artifact["level"]` ya trae el sufijo de combinación para niveles split_by
    (10-12, p.ej. "level_12_daily_item_store__CA_1_FOODS_1" -- ver load_data()), así
    que alcanza para reconstruir el path exacto sin guardar split_values aparte."""
    # Grain real del artifact, no level.grains[0]: un artifact weekly (level_str
    # p.ej. "level_01_weekly_total") reconstruido con "daily" a secas buscaría el
    # parquet en la carpeta equivocada (data/datasets/daily/) y cargaría, si por
    # coincidencia existiera un archivo con ese nombre ahí, el dataset equivocado.
    m_grain = re.search(r"level_\d+_(daily|weekly)", artifact["level"])
    grain = m_grain.group(1) if m_grain else config.LEVELS_BY_ID[artifact["level_id"]].grains[0]
    dataset_path = config.DATASETS / grain / f"dataset_{artifact['level']}.parquet"

    state = load_data(artifact["level_id"], artifact["target"], dataset_path, grain=grain)
    state.features = artifact["features"]
    state.categorical_features = artifact["categorical_features"]
    state.numerical_features = artifact["numerical_features"]
    split_data(state, artifact["target"], test_start=pd.Timestamp(artifact["test_start"]))
    return state.X_test, state.y_test, state.train, state.test
