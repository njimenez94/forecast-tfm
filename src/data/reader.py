import polars as pl
import duckdb


def read_parquet_pl(file) -> pl.DataFrame:
    """Lee un parquet bajando float64→float32 con scan lazy."""
    lf = pl.scan_parquet(file)
    schema = lf.collect_schema()
    casts = [pl.col(c).cast(pl.Float32) for c, t in schema.items() if t == pl.Float64]
    if casts:
        lf = lf.with_columns(casts)
    return lf.collect()


def read_query_str(db_path, sql: str):
    """Ejecuta SQL sobre DuckDB y devuelve un DataFrame pandas."""
    with duckdb.connect(str(db_path)) as con:
        return con.execute(sql).df()
