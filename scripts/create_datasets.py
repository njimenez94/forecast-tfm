"""Construye datasets por nivel de agregación (L1 → L12).

Cada nivel produce parquets en data/processed/ con el esquema:
    dataset_level_{id:02d}_{grain}_{window}_{target}_{name}.parquet

donde {target} es "sales" (venta individual) o "cumN" (suma acumulada de N períodos).
Los lags y rolling windows los genera mlforecast internamente durante el entrenamiento.

Uso:
    python -m scripts.create_datasets                  # genera todos los niveles y ventanas
    python -m scripts.create_datasets --levels 1,9,12  # solo esos niveles
    python -m scripts.create_datasets --counts         # solo series/filas, sin materializar
"""
import argparse
import gc
from datetime import date, timedelta

import duckdb
from loguru import logger

import config
from src.data.base_query import build_base_query, build_cum_query, build_exog_query, count_series_query
from src.data.reader import read_query_str


def _max_date(db_path) -> date:
    row = read_query_str(
        db_path,
        "SELECT MAX(c.date) AS d FROM calendar c "
        "WHERE c.d IN (SELECT DISTINCT d FROM sales_train_evaluation)",
    )
    return row["d"][0]


def _periods(db_path) -> dict[str, int]:
    days = read_query_str(
        db_path, "SELECT COUNT(DISTINCT d) AS n FROM sales_train_evaluation"
    )["n"][0]
    weeks = read_query_str(
        db_path,
        "SELECT COUNT(DISTINCT wm_yr_wk) AS n FROM calendar "
        "WHERE d IN (SELECT DISTINCT d FROM sales_train_evaluation)",
    )["n"][0]
    return {"daily": int(days), "weekly": int(weeks)}


def show_counts(levels, db_path) -> None:
    periods = _periods(db_path)
    logger.info(f"Periodos disponibles: {periods['daily']} días / {periods['weekly']} semanas")
    logger.info(f"{'lvl':>3}  {'nombre':12} {'grain':7} {'series':>8} {'filas':>14}")
    for lvl in levels:
        n = int(read_query_str(db_path, count_series_query(lvl.dims))["n"][0])
        for grain in lvl.grains:
            rows = n * periods[grain]
            logger.info(f"{lvl.id:>3}  {lvl.name:12} {grain:7} {n:>8} {rows:>14,}")


def _build_sql(lvl, grain: str, min_date, cum_n_val: int | None) -> str:
    """Construye el SQL final para el target dado (sales o cumN)."""
    base_sql = build_base_query(lvl.dims, grain)
    exog_sql = build_exog_query(base_sql, grain)
    if min_date is not None:
        exog_sql = f"SELECT * FROM ({exog_sql}) WHERE date >= '{min_date}'"
    if cum_n_val is not None:
        exog_sql = build_cum_query(exog_sql, cum_n_val)
    return exog_sql


def generate_level(db_path, sql: str, out) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"COPY ({sql}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def generate(levels, db_path, processed_dir) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    max_date = _max_date(db_path)

    all_targets = {
        grain: ["sales"] + [f"cum{n}" for n in config.CUM_HORIZONS[grain]]
        for grain in ("daily", "weekly")
    }

    for lvl in levels:
        n_series = int(read_query_str(db_path, count_series_query(lvl.dims))["n"][0])
        for grain in lvl.grains:
            for window_days in config.TRAIN_WINDOW_DAYS[grain]:
                wlabel   = config.window_label(window_days)
                min_date = None if window_days is None else max_date - timedelta(days=window_days)
                for target in all_targets[grain]:
                    cum_n_val = config.cum_n(target)
                    out = config.dataset_level_path(lvl, grain, window_days, target)
                    logger.info(
                        "[L{} {}/{}] {} series | {} | target={} → {}",
                        lvl.id, lvl.name, grain, n_series, wlabel, target, out.name,
                    )
                    sql = _build_sql(lvl, grain, min_date, cum_n_val)
                    generate_level(db_path, sql, out)
                    size_mb = out.stat().st_size / 1_048_576
                    logger.success("  listo  {:.1f} MB", size_mb)
                    gc.collect()


def parse_levels(arg: str | None):
    if not arg:
        return config.LEVELS
    ids = {int(x) for x in arg.split(",")}
    return [lvl for lvl in config.LEVELS if lvl.id in ids]


def main():
    ap = argparse.ArgumentParser(description="Genera datasets por nivel de agregación M5.")
    ap.add_argument("--levels", help="IDs separados por coma, p.ej. 1,9,12. Por defecto: todos.")
    ap.add_argument("--counts", action="store_true",
                    help="Solo mostrar series/filas por nivel, sin materializar.")
    args = ap.parse_args()

    levels = parse_levels(args.levels)
    if args.counts:
        show_counts(levels, config.DB_PATH)
    else:
        generate(levels, config.DB_PATH, config.PROCESSED_DIR)


if __name__ == "__main__":
    main()
