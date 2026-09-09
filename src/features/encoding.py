"""Target encoding causal: rolling mean/std por item_id/dept_id/cat_id (cross-serie,
venta promedio de una serie cualquiera que comparte ese item/dept/cat) + media
expandiendo por series_id condicionada a weekend/snap/event (estacionalidad propia
de cada serie). Debe correr después de add_event_features: usa 'sales' ya limpio del
zero-out de is_store_closed.

_daily se agrega con mean(), no sum(): item/dept/cat deben quedar en la misma unidad
("venta de una serie típica ese día") para que _bayesian_shrink pueda mezclarlos --
con sum(), dept (que agrega decenas de items) queda en una escala muchísimo más alta
que item, y el shrink dispara el resultado hacia ese total en vez de promediarlo.
"""
import polars as pl

# Ventana (días) del target encoding rolling por item/dept/cat -- mismo horizonte que
# el resto del pipeline (ver HORIZON en config/features.py).
_WINDOW = 28

# Pseudo-observaciones del prior (nivel jerárquico superior) en el shrinkage
# Bayesiano -- ver _bayesian_shrink. Knob de calibración: más alto = más shrink
# incluso con ventana llena; 10 es ~1/3 de _WINDOW, shrink moderado.
_SHRINK_PRIOR_WEIGHT = 10


def _conditional_expanding_mean(flag_col: str) -> pl.Expr:
    """Media expandiendo de sales del propio series_id, condicionada a flag_col (solo
    ocurrencias pasadas con el mismo valor de flag_col: p.ej. media histórica de venta
    en fin de semana vs. entre semana). Shift para no leakear el día actual -- NaN
    hasta la primera repetición pasada de ese valor de flag.
    """
    grp = ["series_id", flag_col]
    cum_sum_prev = pl.col("sales").cum_sum().over(grp).shift(1).over(grp)
    cum_cnt_prev = pl.col("sales").cum_count().over(grp).shift(1).over(grp)
    return (cum_sum_prev / cum_cnt_prev).cast(pl.Float32)


def _rolling_target_encoding(df: pl.DataFrame, group_col: str, window: int = _WINDOW) -> pl.DataFrame:
    """Añade te_<grupo>_mean/std/n: rolling mean/std/nº-de-días de la venta diaria
    promedio entre las series que comparten group_col (p.ej. item_id: venta media
    de las series de ese item, cada día -- ver nota de módulo sobre por qué mean() y
    no sum()), shift(1) para no leakear el día actual. te_*_n (días reales dentro de
    la ventana, tope `window`) alimenta _bayesian_shrink y se descarta después en
    add_encoding_features. Cuando group_col es constante 'TOTAL' en el nivel (dim que
    no varía, ver build_base_query), degenera a una única serie global -- válido,
    solo redundante con otras features de nivel.
    """
    key = group_col.removesuffix("_id")
    daily = (
        df.group_by([group_col, "date"])
        .agg(pl.col("sales").mean().alias("_daily"))
        .sort([group_col, "date"])
        .with_columns(
            pl.col("_daily").rolling_mean(window_size=window, min_samples=1)
                .over(group_col).shift(1).over(group_col)
                .cast(pl.Float32).alias(f"te_{key}_mean"),
            pl.col("_daily").rolling_std(window_size=window, min_samples=2)
                .over(group_col).shift(1).over(group_col)
                .cast(pl.Float32).alias(f"te_{key}_std"),
            # pl.lit(1) no sirve aquí: al ser un literal escalar, rolling_sum no lo
            # trata como una columna real y no cuenta filas por grupo/ventana --
            # is_not_null() sobre _daily (siempre no-nula, panel denso) sí.
            pl.col("_daily").is_not_null().cast(pl.Int32)
                .rolling_sum(window_size=window, min_samples=1)
                .over(group_col).shift(1).over(group_col)
                .alias(f"_te_{key}_n"),
        )
        .select([group_col, "date", f"te_{key}_mean", f"te_{key}_std", f"_te_{key}_n"])
    )
    return df.join(daily, on=[group_col, "date"], how="left")


def _bayesian_shrink(child_mean: str, child_n: str, parent_mean: str, prior_weight: float = _SHRINK_PRIOR_WEIGHT) -> pl.Expr:
    """Shrinkage empírico-Bayesiano hacia el nivel jerárquico superior:
    (n * child_mean + prior_weight * parent_mean) / (n + prior_weight). Con pocas
    observaciones propias (child_n bajo -- item nuevo o de baja rotación) el
    resultado tira hacia el padre (dept/cat, con mucha más muestra); con la ventana
    llena, converge a una mezcla estable ~child_n/(child_n+prior_weight) propia.
    """
    n = pl.col(child_n)
    return ((n * pl.col(child_mean) + prior_weight * pl.col(parent_mean)) / (n + prior_weight)).cast(pl.Float32)


def add_encoding_features(df: pl.DataFrame) -> pl.DataFrame:
    for col in ("item_id", "dept_id", "cat_id"):
        df = _rolling_target_encoding(df, col)
    # Orden jerárquico: dept se corrige contra cat primero, luego item contra el
    # dept ya corregido (así item hereda transitivamente el ajuste de cat).
    df = df.with_columns(_bayesian_shrink("te_dept_mean", "_te_dept_n", "te_cat_mean").alias("te_dept_mean"))
    df = df.with_columns(_bayesian_shrink("te_item_mean", "_te_item_n", "te_dept_mean").alias("te_item_mean"))
    df = df.drop("_te_item_n", "_te_dept_n", "_te_cat_n")
    return df.with_columns(
        _conditional_expanding_mean("is_weekend").alias("te_weekend_mean"),
        _conditional_expanding_mean("snap").alias("te_snap_mean"),
        _conditional_expanding_mean("has_event").alias("te_event_mean"),
    )
