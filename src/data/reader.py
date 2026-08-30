import polars as pl
import duckdb


def read_parquet_pl(file, filter_expr: pl.Expr | None = None) -> pl.DataFrame:
    """Lee un parquet bajando float64→float32 con scan lazy. filter_expr (opcional)
    se empuja al scan antes de collect(), para leer solo una partición sin
    materializar el resto (ver scripts/build_datasets.py, niveles de alta cardinalidad)."""
    lf = pl.scan_parquet(file)
    schema = lf.collect_schema()
    casts = [pl.col(c).cast(pl.Float32) for c, t in schema.items() if t == pl.Float64]
    if casts:
        lf = lf.with_columns(casts)
    if filter_expr is not None:
        lf = lf.filter(filter_expr)
    return lf.collect()


def read_query_str(db_path, sql: str):
    """Ejecuta SQL sobre DuckDB y devuelve un DataFrame pandas."""
    with duckdb.connect(str(db_path)) as con:
        return con.execute(sql).df()


def count_periods(db_path) -> dict[str, int]:
    """Nº de días/semanas distintos en sales_train_evaluation -- usado para estimar
    filas por nivel antes de materializar (ver scripts/process_data.py --counts)."""
    days = read_query_str(db_path, "SELECT COUNT(DISTINCT d) AS n FROM sales_train_evaluation")["n"][0]
    weeks = read_query_str(
        db_path,
        "SELECT COUNT(DISTINCT wm_yr_wk) AS n FROM calendar "
        "WHERE d IN (SELECT DISTINCT d FROM sales_train_evaluation)",
    )["n"][0]
    return {"daily": int(days), "weekly": int(weeks)}


def write_query_parquet(db_path, sql: str, out, *, memory_limit: str = "3GB", threads: int = 4) -> None:
    """Ejecuta sql sobre DuckDB y escribe el resultado directo a parquet (COPY TO, sin
    traer el resultado a Python). Límites de memoria conservadores: el default de
    DuckDB (80% RAM) no cuenta el resto de procesos del host y provoca OOM-kill
    (Error 137) en niveles grandes (L12)."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA threads={threads}")
        con.execute(f"COPY ({sql}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
