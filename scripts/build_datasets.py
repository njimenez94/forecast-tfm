"""Aplica src.features sobre cada dataset base (data/processed/) y guarda
el resultado -- dataframe final, con lags/rolling/momentum ya materializados, listo
para entrenar cualquier modelo sin pasar por mlforecast -- en artifacts/datasets/.

Uso:
    python -m scripts.build_datasets                  # niveles activos (config.ACTIVE_LEVEL_IDS)
    python -m scripts.build_datasets --levels 1,9,12   # solo esos niveles
"""
import argparse
import gc
import warnings

import humanize
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

import config
from src.data.reader import read_parquet_pl
from src.logging_setup import configure_logging
from src.data.split import parse_level_file
from src.features import build_dataset

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# Por encima de este nº de filas, procesar el nivel partición a partición (store_id
# o, si solo hay un store, state_id) en vez de cargar el nivel entero: a partir de
# item_state (~18M filas) el pico de RAM -- float64 intermedio antes del downcast +
# la copia que hace mlforecast.preprocess -- satura la memoria disponible. Cada
# partición procesada por separado pesa lo mismo que un nivel "item" (~6M filas),
# que ya se sabe que corre sin problema.
CHUNK_ROW_THRESHOLD = 10_000_000


def _partition_col(file) -> str | None:
    """store_id si tiene más de un valor (nivel item_store), si no state_id (nivel
    item_state), si no None (no hace falta particionar)."""
    lf = pl.scan_parquet(file)
    cols = lf.collect_schema().names()
    for col in ("store_id", "state_id"):
        if col in cols and lf.select(pl.col(col).n_unique()).collect().item() > 1:
            return col
    return None


def _process_split(file, level, grain: str, static_cols: list[str]) -> None:
    """Un parquet independiente por cada combinación de level.split_by (p.ej.
    store_id x dept_id en nivel 12): datasets y modelos entrenables por separado
    en vez de un único dataset gigante para todo el nivel."""
    cols = list(level.split_by)
    combos = pl.scan_parquet(file).select(cols).unique().sort(cols).collect().rows()
    logger.info("  separando por {} ({} datasets)", cols, len(combos))

    for i, combo in enumerate(combos, 1):
        split_values = dict(zip(cols, combo))
        filter_expr = pl.all_horizontal([pl.col(c) == v for c, v in split_values.items()])
        final = build_dataset(read_parquet_pl(file, filter_expr=filter_expr), grain, static_cols)
        out = config.featured_level_path(level, grain, split_values)
        final.to_parquet(out, compression="zstd", index=False)
        logger.success("  [{}/{}] [{}] {:.1f} MB  ({} filas)", i, len(combos), "/".join(map(str, combo)),
                       out.stat().st_size / 1_048_576, humanize.intcomma(final.shape[0]))
        del final
        gc.collect()


def main():
    ap = argparse.ArgumentParser(
        description="Genera el dataset final (features + lags/rolling) en artifacts/datasets/."
    )
    ap.add_argument("--levels", help="IDs separados por coma, p.ej. 1,9,12. "
                    "Por defecto: config.ACTIVE_LEVEL_IDS.")
    args = ap.parse_args()
    level_ids = {int(x) for x in args.levels.split(",")} if args.levels else set(config.ACTIVE_LEVEL_IDS)

    files = sorted(config.PROCESSED_DIR.glob("*/level_*.parquet"))
    if not files:
        logger.error("No hay parquets en {}. Ejecuta primero: make process-data", config.PROCESSED_DIR)
        return

    for grain in ("daily", "weekly"):
        (config.DATASETS / grain).mkdir(parents=True, exist_ok=True)

    for file in files:
        parsed = parse_level_file(file)
        if parsed is None:
            logger.warning("SKIP legacy: {}", file.name)
            continue
        level, grain = parsed
        if level_ids is not None and level.id not in level_ids:
            continue

        static_cols = [d for d in level.dims if d not in config.EXCLUDE_AS_STATIC]

        if level.split_by:
            logger.info("[L{} {}/{}] {}", level.id, level.name, grain, file.name)
            _process_split(file, level, grain, static_cols)
            continue

        out = config.featured_level_path(level, grain)
        logger.info("[L{} {}/{}] {} → {}", level.id, level.name, grain, file.name, out.name)

        row_count = pl.scan_parquet(file).select(pl.len()).collect().item()
        part_col = _partition_col(file) if row_count > CHUNK_ROW_THRESHOLD else None

        if part_col is None:
            final = build_dataset(read_parquet_pl(file), grain, static_cols)
            final.to_parquet(out, compression="zstd", index=False)
            n_rows, n_cols = final.shape
        else:
            values = pl.scan_parquet(file).select(pl.col(part_col)).unique().collect()[part_col].to_list()
            logger.info("  particionando por {} ({} partes, evita OOM)", part_col, len(values))
            writer = None
            n_rows = n_cols = 0
            for value in values:
                final = build_dataset(read_parquet_pl(file, filter_expr=pl.col(part_col) == value), grain, static_cols)
                table = pa.Table.from_pandas(final, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema, compression="zstd")
                writer.write_table(table)
                n_rows += final.shape[0]
                n_cols = final.shape[1]
                del final, table
                gc.collect()
            writer.close()

        logger.success("  listo  {:.1f} MB  ({} cols, {} filas)",
                       out.stat().st_size / 1_048_576, n_cols, humanize.intcomma(n_rows))


if __name__ == "__main__":
    configure_logging("build_datasets")
    main()
