"""Construye datasets por nivel de agregación (L1 → L12): data/processed/.

Cada nivel produce un parquet en data/processed/ con el esquema:
    level_{id:02d}_{grain}_{name}.parquet

con la columna "sales" (venta individual). Historia completa, sin recorte por
ventana de entrenamiento (eso se explora en tuning). Los lags y rolling windows los
genera mlforecast internamente durante el entrenamiento.

Siguiente paso del pipeline: scripts/build_datasets.py añade features derivadas
sobre este dataset y guarda el resultado en artifacts/datasets/.

Uso:
    python -m scripts.process_data                  # niveles activos (config.ACTIVE_LEVEL_IDS)
    python -m scripts.process_data --levels 1,9,12  # solo esos niveles
    python -m scripts.process_data --counts         # solo series/filas, sin materializar
"""
import argparse
import gc

import humanize
from loguru import logger

import config
from src.data.base_query import build_level_query, count_series_query
from src.data.reader import count_periods, read_query_str, write_query_parquet
from src.logging_setup import configure_logging


def show_counts(levels, db_path) -> None:
    periods = count_periods(db_path)
    logger.info(
        "Periodos disponibles: {} días / {} semanas",
        humanize.intcomma(periods["daily"]), humanize.intcomma(periods["weekly"]),
    )
    for lvl in levels:
        n = int(read_query_str(db_path, count_series_query(lvl.dims, lvl.filters))["n"][0])
        for grain in lvl.grains:
            rows = n * periods[grain]
            logger.info(
                "[L{} {}/{}] {} series × {} {} → {} filas",
                lvl.id, lvl.name, grain,
                humanize.intcomma(n), humanize.intcomma(periods[grain]), grain,
                humanize.intcomma(rows),
            )


def generate(levels, db_path, processed_dir) -> None:
    for grain in ("daily", "weekly"):
        (processed_dir / grain).mkdir(parents=True, exist_ok=True)
    periods = count_periods(db_path)

    for lvl in levels:
        n_series = int(read_query_str(db_path, count_series_query(lvl.dims, lvl.filters))["n"][0])
        for grain in lvl.grains:
            out = config.dataset_level_path(lvl, grain)
            n_periods = periods[grain]
            n_rows = n_series * n_periods
            logger.info("[L{} {}/{}] → {}", lvl.id, lvl.name, grain, out.name)
            if lvl.filters:
                logger.info("  filtro: {}", lvl.filters)
            logger.info(
                "  {} series × {} {}", humanize.intcomma(n_series), humanize.intcomma(n_periods), grain,
            )
            logger.info("  {} filas (aprox.)", humanize.intcomma(n_rows))
            sql = build_level_query(lvl.dims, grain, lvl.filters)
            write_query_parquet(db_path, sql, out)
            logger.success("  listo  {}", humanize.naturalsize(out.stat().st_size, binary=True))
            gc.collect()


def parse_levels(arg: str | None):
    if not arg:
        return config.ACTIVE_LEVELS
    ids = {int(x) for x in arg.split(",")}
    return [lvl for lvl in config.LEVELS if lvl.id in ids]


def main():
    ap = argparse.ArgumentParser(description="Genera datasets por nivel de agregación M5.")
    ap.add_argument("--levels", help="IDs separados por coma, p.ej. 1,9,12. "
                    "Por defecto: config.ACTIVE_LEVEL_IDS.")
    ap.add_argument("--counts", action="store_true",
                    help="Solo mostrar series/filas por nivel, sin materializar.")
    args = ap.parse_args()

    levels = parse_levels(args.levels)
    if args.counts:
        show_counts(levels, config.DB_PATH)
    else:
        generate(levels, config.DB_PATH, config.PROCESSED_DIR)


if __name__ == "__main__":
    configure_logging("process_data")
    main()
