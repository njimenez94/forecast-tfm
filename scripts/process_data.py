"""Construye datasets por nivel de agregación (L1 → L12): data/processed/.

Cada nivel produce un parquet en data/processed/ con el esquema:
    level_{id:02d}_{grain}_{name}.parquet

con la columna "sales" (venta individual) y columnas "cumN" (suma acumulada forward
de N períodos) para cada N en config.CUM_HORIZONS[grain] -- el target se elige en
train time por nombre de columna, no por archivo. Historia completa, sin recorte por
ventana de entrenamiento (eso se explora en tuning). Los lags y rolling windows los
genera mlforecast internamente durante el entrenamiento.

Siguiente paso del pipeline: scripts/build_datasets.py añade features derivadas
sobre este dataset y guarda el resultado en artifacts/datasets/.

Uso:
    python -m scripts.process_data                  # genera todos los niveles
    python -m scripts.process_data --levels 1,9,12  # solo esos niveles
    python -m scripts.process_data --counts         # solo series/filas, sin materializar
"""
import argparse
import gc

import duckdb
import humanize
from loguru import logger

import config
from src.data.base_query import build_base_query, build_cum_query, build_exog_query, count_series_query
from src.data.reader import read_query_str


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
    logger.info(
        "Periodos disponibles: {} días / {} semanas",
        humanize.intcomma(periods["daily"]), humanize.intcomma(periods["weekly"]),
    )
    for lvl in levels:
        n = int(read_query_str(db_path, count_series_query(lvl.dims))["n"][0])
        for grain in lvl.grains:
            rows = n * periods[grain]
            logger.info(
                "[L{} {}/{}] {} series × {} {} → {} filas",
                lvl.id, lvl.name, grain,
                humanize.intcomma(n), humanize.intcomma(periods[grain]), grain,
                humanize.intcomma(rows),
            )


def _build_sql(lvl, grain: str) -> str:
    """Construye el SQL final: sales + una columna cumN por horizonte configurado."""
    base_sql = build_base_query(lvl.dims, grain)
    exog_sql = build_exog_query(base_sql, grain)
    return build_cum_query(exog_sql, config.CUM_HORIZONS[grain])


def generate_level(db_path, sql: str, out) -> None:
    with duckdb.connect(str(db_path)) as con:
        # Límite conservador: el default (80% RAM) no cuenta el resto de procesos del
        # host y provoca OOM-kill (Error 137) del proceso en niveles grandes (L12).
        con.execute("PRAGMA memory_limit='3GB'")
        con.execute("PRAGMA threads=4")
        con.execute(f"COPY ({sql}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def generate(levels, db_path, processed_dir) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    periods = _periods(db_path)

    for lvl in levels:
        n_series = int(read_query_str(db_path, count_series_query(lvl.dims))["n"][0])
        for grain in lvl.grains:
            out = config.dataset_level_path(lvl, grain)
            n_periods = periods[grain]
            n_rows = n_series * n_periods
            logger.info("[L{} {}/{}] → {}", lvl.id, lvl.name, grain, out.name)
            logger.info(
                "  {} series × {} {}", humanize.intcomma(n_series), humanize.intcomma(n_periods), grain,
            )
            logger.info("  {} filas (aprox.)", humanize.intcomma(n_rows))
            sql = _build_sql(lvl, grain)
            generate_level(db_path, sql, out)
            logger.success("  listo  {}", humanize.naturalsize(out.stat().st_size, binary=True))
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
