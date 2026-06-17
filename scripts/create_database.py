"""Descomprime el zip y crea la base de datos."""
import zipfile
from loguru import logger
import config

from pathlib import Path
import duckdb

SQL_PATH = "queries/create_database.sql"

raw_dir = config.RAW_DIR
raw_zip = config.RAW_ZIP
db_path= config.DB_PATH

def extract_files():
    logger.info("Comienza extraccion de datos...")
    if not raw_zip.exists():
        raise FileNotFoundError(f"No se encontró el zip en {raw_zip}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Descomprimiendo {raw_zip.name} en {raw_dir}...")
    with zipfile.ZipFile(raw_zip) as zf:
        for name in zf.namelist():
            logger.info(f"  -> {name}")
        zf.extractall(raw_dir)

def load_files():
    logger.info("Comienza creación de base de datos...")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    sql = Path(SQL_PATH).read_text()
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    logger.info(f"Ejecutando {SQL_PATH} ({len(statements)} statements)...")
    with duckdb.connect(db_path) as con:
        for stmt in statements:
            con.execute(stmt)

        logger.info(f"Base de datos lista en '{db_path}'")
        for (table,) in con.execute("SHOW TABLES").fetchall():
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  - {table}: {count:,} filas")

def main():
    extract_files()
    load_files()

if __name__ == "__main__":
    main()