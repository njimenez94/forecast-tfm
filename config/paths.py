"""Rutas del proyecto: datos crudos, base de datos y artefactos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
RAW_ZIP = ROOT / "doc" / "m5-forecasting-accuracy.zip"
DB_PATH = ROOT / "data" / "m5.db"
CREATE_DATABASE_QUERY = ROOT / "queries" / "create_database.sql"

PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "output"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"


def window_label(days: int | None) -> str:
    if days is None:
        return "wmax"
    return f"w{days // 365}y"


def dataset_level_path(level, grain: str, window_days: int | None, target: str = "sales") -> Path:
    return PROCESSED_DIR / f"dataset_level_{level.id:02d}_{grain}_{window_label(window_days)}_{target}_{level.name}.parquet"
