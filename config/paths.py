"""Rutas del proyecto: datos crudos, base de datos y artefactos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
RAW_ZIP = ROOT / "backup" / "m5-forecasting-accuracy.zip"
DB_PATH = ROOT / "data" / "m5.db"
CREATE_DATABASE_QUERY = ROOT / "queries" / "create_database.sql"

# data/processed/: extracción SQL por nivel (sales, cumN, precio, calendario).
# Generado por scripts/process_data.py (make process-data).
PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "output"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# artifacts/datasets/: data/processed/ + features derivadas (src/features/engineer.py),
# listo para entrenar. Generado por scripts/build_datasets.py (make build-datasets).
DATASETS = ROOT / "data" / "datasets"


def dataset_level_path(level, grain: str) -> Path:
    return PROCESSED_DIR / f"level_{level.id:02d}_{grain}_{level.name}.parquet"


def featured_level_path(level, grain: str, split_values: dict | None = None) -> Path:
    """split_values: valores de level.split_by (p.ej. {"store_id": "CA_3", "dept_id":
    "FOODS_3"} en nivel 12) -- un parquet independiente por combinación, ver
    scripts/build_datasets.py."""
    suffix = ""
    if split_values:
        suffix = "__" + "_".join(str(split_values[c]) for c in level.split_by)
    return DATASETS / f"dataset_level_{level.id:02d}_{grain}_{level.name}{suffix}.parquet"
