import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

import config


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
        # test = df >= test_start (sin cota superior), incluye last_date: intervalo
        # cerrado en ambos extremos, a diferencia de train/valid que son semiabiertos.
        return (self.last_date - self.test_start).days + 1

    def log_summary(self) -> None:
        # último día real de train/valid (no valid_start/test_start): esos son el
        # borde exclusivo del siguiente split, mostrarlos acá se lee como fechas
        # duplicadas entre splits aunque las filas no se solapen.
        train_last = self.train["date"].max()
        valid_last = self.valid["date"].max()
        logger.info(f"Train : {self.first_date:%Y-%m-%d} to {train_last:%Y-%m-%d} ({self.train_days:,} days, {len(self.train):,} rows)")
        logger.info(f"Valid : {self.valid_start:%Y-%m-%d} to {valid_last:%Y-%m-%d} ({self.valid_days:,} days, {len(self.valid):,} rows)")
        logger.info(f"Test  : {self.test_start:%Y-%m-%d} to {self.last_date:%Y-%m-%d} ({self.test_days:,} days, {len(self.test):,} rows)")


def date_split(df: pd.DataFrame, valid_days: int, test_days: int, date_col: str = "date",
                test_start: pd.Timestamp | None = None, tail_reserve_days: int = 0) -> DateSplit:
    """Split temporal simple train/valid/test por fecha, sobre un dataframe ya con
    features. Los últimos `test_days` quedan como test, los `valid_days` anteriores
    como validación, y todo lo previo como train.

    `test_start` fija el borde del split en vez de derivarlo de `last_date`: lo usa
    reconstruct_test_data() (src/data/dataset.py) para reproducir exactamente el
    split de un artifact ya entrenado, aunque el parquet se haya regenerado después.

    `tail_reserve_days`: descarta las últimas N filas de date antes de partir train/
    valid/test, para el caso general de un target con cola incompleta al final de
    cada serie. Se aplica también cuando `test_start` viene fijo (reconstruct): sin
    esto, `test` (sin cota superior) volvería a incluir esa cola NULL."""
    if tail_reserve_days:
        df = df[df[date_col] <= df[date_col].max() - pd.Timedelta(days=tail_reserve_days)]
    first_date = df[date_col].min()
    last_date = df[date_col].max()
    if test_start is None:
        # test = df >= test_start (sin cota superior) incluye last_date, así que el
        # intervalo es cerrado en ambos extremos: para que abarque exactamente
        # test_days días hay que restar (test_days - 1).
        test_start = last_date - pd.DateOffset(days=test_days - 1)
    valid_start = test_start - pd.DateOffset(days=valid_days)

    train = df[df[date_col] < valid_start]
    valid = df[(df[date_col] >= valid_start) & (df[date_col] < test_start)]
    test = df[df[date_col] >= test_start]

    return DateSplit(
        train=train, valid=valid, test=test,
        first_date=first_date, last_date=last_date,
        valid_start=valid_start, test_start=test_start,
    )


_LAG_PREFIXES = ("lag", "rolling_", "expanding_", "seasonal_", "momentum_")


def _lag_anchor(col: str) -> int | None:
    """Lag base de una columna generada por mlforecast (lag7, rolling_mean_lag28_
    window_size7, ...), o None si no es una feature derivada del target."""
    matches = re.findall(r"lag(\d+)", col)
    return min(int(m) for m in matches) if matches else None


def mask_horizon_leakage(X: pd.DataFrame, dates: pd.Series, origin: pd.Timestamp | np.ndarray, grain: str) -> pd.DataFrame:
    """NaN-ea, fila por fila, los lags/rolling/expanding/seasonal cuyo anchor sea
    menor al paso h = (fecha - origin) en períodos de `grain`.

    Un pronóstico real se hace de una sola tirada desde `origin` (el último día
    conocido), sin actualizar con datos reales intermedios: para la fila que está
    h períodos después de origin, ningún lag menor a h existe todavía -- lag_k
    solo es seguro si k >= h. LightGBM/XGBoost manejan NaN nativamente, así que
    enmascarar (no descartar columnas ni filas) conserva el máximo de señal legítima
    disponible para cada fila.

    `origin` puede ser un único Timestamp (todo el split comparte origen) o un
    array con un origen por fila (ver `block_origins`, para revalidar el horizonte
    dentro de bloques consecutivos en vez de un único origen para todo el split).
    """
    period_days = 7 if grain == "weekly" else 1
    h = ((dates - origin).dt.days // period_days).to_numpy()

    X = X.copy()
    for col in X.columns:
        if not col.startswith(_LAG_PREFIXES):
            continue
        anchor = _lag_anchor(col)
        if anchor is not None:
            X.loc[h > anchor, col] = np.nan
    return X


def block_origins(dates: pd.Series, ref_start: pd.Timestamp, block_periods: int, grain: str) -> np.ndarray:
    """Origen por fila para revalidar horizonte dentro de bloques consecutivos de
    `block_periods` (unidades de `grain`) que arrancan en `ref_start` y se repiten
    hasta cubrir `dates`.

    Simula un pronóstico que se refresca cada `block_periods` con datos reales ya
    observados hasta el día anterior a cada bloque -- en vez de un único origen fijo
    para todo el split, que fuerza a enmascarar casi toda la señal reciente cuando el
    split es mucho más largo que el horizonte real de despliegue (ver
    mask_horizon_leakage y config.VALID_PERIODS)."""
    period_days = 7 if grain == "weekly" else 1
    block_days = block_periods * period_days
    offset_days = ((dates - ref_start).dt.days // block_days) * block_days
    block_start = ref_start + pd.to_timedelta(offset_days, unit="D")
    return (block_start - pd.Timedelta(days=period_days)).to_numpy()


def valid_blocks(valid_start: pd.Timestamp, test_start: pd.Timestamp, block_periods: int,
                 grain: str) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """Bloques consecutivos de `block_periods` que cubren [valid_start, test_start),
    para evaluar métricas bloque por bloque (en vez de agregadas sobre todo el
    período). Toma los timestamps sueltos en vez de un DateSplit para poder
    descartar train/valid/test (pesados) y quedarse solo con los bordes. Devuelve
    (n, start, end) con end exclusivo; el último bloque puede ser más corto si el
    total no es múltiplo exacto de block_periods."""
    period_days = 7 if grain == "weekly" else 1
    block_days = block_periods * period_days
    blocks = []
    start, n = valid_start, 0
    while start < test_start:
        end = min(start + pd.DateOffset(days=block_days), test_start)
        blocks.append((n, start, end))
        start, n = end, n + 1
    return blocks


def build_feature_matrices(df: pd.DataFrame, split: DateSplit, features: list[str],
                           categorical_features: list[str], target: str, grain: str):
    """Arma X/y para train/valid/test y unifica las categorías de las columnas
    categóricas a partir del dataset completo: evita que cada split termine con un
    set de categorías distinto (rompe modelos que las validan, p. ej. XGBoost).

    X_valid/X_test se enmascaran por bloques consecutivos de config.VALID_PERIODS[grain]
    / config.TEST_PERIODS[grain] (ver block_origins): cada bloque revalida el horizonte
    con su propio origen, como si el pronóstico se refrescara cada VALID_PERIODS con
    datos reales ya observados. Si split.valid dura exactamente un bloque (el caso
    estándar) esto da el mismo resultado que un único origen para todo el split; si
    dura más (p.ej. config.valid_year_days(grain), el último año) evita enmascarar
    de más la señal reciente en la mayoría de las filas."""
    X_train = split.train[features].copy()
    X_valid = split.valid[features].copy()
    X_test = split.test[features].copy()

    origin_valid = block_origins(split.valid["date"], split.valid_start, config.VALID_PERIODS[grain], grain)
    origin_test = block_origins(split.test["date"], split.test_start, config.TEST_PERIODS[grain], grain)
    X_valid = mask_horizon_leakage(X_valid, split.valid["date"], origin_valid, grain)
    X_test = mask_horizon_leakage(X_test, split.test["date"], origin_test, grain)

    for col in categorical_features:
        categories = df[col].astype("category").cat.categories
        for X in (X_train, X_valid, X_test):
            X[col] = pd.Categorical(X[col], categories=categories)

    y_train = split.train[target]
    y_valid = split.valid[target]
    y_test = split.test[target]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def rolling_cv_folds(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
                     train_dates: pd.Series, valid_start: pd.Timestamp, grain: str,
                     n_folds: int) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """Folds rolling-origin sobre train+valid (todo lo anterior a test), para elegir
    un n_estimators robusto antes de reentrenar el modelo final con todos los datos
    (ver scripts/train_dataset.py fit_final_model).

    El fold más reciente es exactamente (X_train, X_valid) ya armados por
    build_feature_matrices; los `n_folds - 1` anteriores retroceden desde valid_start
    en bloques de config.VALID_PERIODS[grain], recortando la cola de X_train, y
    enmascaran ese recorte con mask_horizon_leakage/block_origins (mismo criterio que
    X_valid) para que el n_estimators elegido no dependa de señal que un pronóstico
    real no tendría en ese fold.

    Devuelve una lista de (X_tr, y_tr, X_val, y_val) del fold más antiguo al más
    reciente; con n_folds=1 devuelve solo el fold actual (X_train, X_valid).
    """
    period_days = 7 if grain == "weekly" else 1
    block_days = config.VALID_PERIODS[grain] * period_days

    folds = []
    for i in range(n_folds - 1, 0, -1):
        fold_valid_start = valid_start - pd.Timedelta(days=i * block_days)
        fold_valid_end = fold_valid_start + pd.Timedelta(days=block_days)
        train_mask = train_dates < fold_valid_start
        val_mask = (train_dates >= fold_valid_start) & (train_dates < fold_valid_end)

        val_dates = train_dates[val_mask]
        origin = block_origins(val_dates, fold_valid_start, config.VALID_PERIODS[grain], grain)
        X_val = mask_horizon_leakage(X_train[val_mask], val_dates, origin, grain)

        folds.append((X_train[train_mask], y_train[train_mask], X_val, y_train[val_mask]))

    folds.append((X_train, y_train, X_valid, y_valid))
    return folds
