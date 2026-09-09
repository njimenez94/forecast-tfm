"""Corrida mínima de punta a punta para validar que el pipeline no se rompió:
perfil fast, un solo nivel/target barato, sin exportar artifact (no pisa los
artifacts reales ya entrenados). No mide calidad de modelo -- para eso,
`python -m scripts.train_dataset` con perfil optimized. Correr antes/después de
cada fase del refactor y comparar el log (split, WAPE/WRMSSE de bench y final).

    uv run python -m scripts.smoke_check
"""
from dataclasses import asdict, replace

import config
from scripts.train_dataset.config import CFG
from scripts.train_dataset.pipeline import run_pipeline
from src.logging_setup import configure_logging


def main():
    profile = config.TRAINING_PROFILES["fast"]
    profile_overrides = {k: v for k, v in asdict(profile).items() if k != "name"}
    cfg = replace(CFG, level_id=1, grain="daily",
                  save_artifact=False, **profile_overrides)
    run_pipeline(cfg)


if __name__ == "__main__":
    configure_logging("smoke_check")
    main()
