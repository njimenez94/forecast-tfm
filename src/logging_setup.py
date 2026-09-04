"""Logging centralizado con Loguru: cada entrypoint (scripts/*.py, api/) escribe a
su propia carpeta bajo logs/, un archivo nuevo por ejecución (nombrado con timestamp).
La consola sigue recibiendo el sink por defecto de Loguru; esto solo agrega el sink
a archivo."""
from datetime import datetime

from loguru import logger

import config


def configure_logging(name: str) -> None:
    log_dir = config.LOGS_DIR / name
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(
        log_dir / f"{timestamp}.log",
        level="DEBUG",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
