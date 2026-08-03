"""Aplica src.features.engineer sobre cada dataset base (data/processed/) y guarda
el resultado -- dataframe final, con lags/rolling/momentum ya materializados, listo
para entrenar cualquier modelo sin pasar por mlforecast -- en artifacts/datasets/.

Uso:
    python -m scripts.build_datasets                  # todos los niveles
    python -m scripts.build_datasets --levels 1,9,12   # solo esos niveles
"""
import argparse
import warnings

from loguru import logger

import config
from src.data.reader import read_parquet_pl
from src.data.split import parse_level_file
from src.features.engineer import add_features, add_lag_features

warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def main():
    ap = argparse.ArgumentParser(
        description="Genera el dataset final (features + lags/rolling) en artifacts/datasets/."
    )
    ap.add_argument("--levels", help="IDs separados por coma, p.ej. 1,9,12. Por defecto: todos.")
    args = ap.parse_args()
    level_ids = {int(x) for x in args.levels.split(",")} if args.levels else None

    files = sorted(config.PROCESSED_DIR.glob("level_*.parquet"))
    if not files:
        logger.error("No hay parquets en {}. Ejecuta primero: make process-data", config.PROCESSED_DIR)
        return

    config.FEATURED_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        parsed = parse_level_file(file)
        if parsed is None:
            logger.warning("SKIP legacy: {}", file.name)
            continue
        level, grain = parsed
        if level_ids is not None and level.id not in level_ids:
            continue

        out = config.featured_level_path(level, grain)
        logger.info("[L{} {}/{}] {} → {}", level.id, level.name, grain, file.name, out.name)
        df = add_features(read_parquet_pl(file)).to_pandas()
        static_cols = [d for d in level.dims if d in df.columns and d not in config.EXCLUDE_AS_STATIC]
        final = add_lag_features(df, grain, static_cols)
        final.to_parquet(out, compression="zstd", index=False)
        logger.success("  listo  {:.1f} MB  ({} cols, {} filas)",
                       out.stat().st_size / 1_048_576, final.shape[1], final.shape[0])


if __name__ == "__main__":
    main()
