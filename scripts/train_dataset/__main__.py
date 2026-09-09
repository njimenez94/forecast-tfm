"""Entrena un modelo para un nivel/target, replicando el flujo de
notebooks/02_model.ipynb: split temporal, comparación de modelos base (naive,
estadísticos clásicos), bench de modelos ML (LightGBM/XGBoost/CatBoost/HistGB/Ridge,
cada uno con hiperparámetros default + early stopping, sin Optuna), selección del
modelo "ganador" (mejor familia tree-based del bench), selección de features
(permutation importance + backward elimination) y explicación SHAP sobre el ganador,
tuning con Optuna (única ronda del pipeline, post-feature-selection, solo sobre el
ganador) y ajuste del modelo final.

El pipeline está separado en fases activables/desactivables (`Config.run_*`, ver
config.py), para poder saltear las que no hacen falta en una corrida dada (p.ej.
iterar sobre selección de features sin repetir el bench de modelos, o reentrenar el
modelo final con los hiperparámetros del bench sin correr Optuna de nuevo).
`run_bench_ml` (benchmarking.py) nunca tunea con Optuna (hiperparámetros fijos +
early stopping, rápido a propósito); `run_optuna` controla la única ronda de Optuna
del pipeline (refinement.py). Estas fases + el presupuesto de Optuna vienen
agrupadas en perfiles (`--profile`, ver config/training.py): `fast` (smoke-test),
`moderate` (primera versión en producción, sin las fases caras de
interpretabilidad) y `optimized` (pipeline completo, default de `Config` -- para
las métricas finales del TFM). Sin --profile se usa la constante `PROFILE` (ver
config.py) -- editarla ahí para cambiar el default sin tener que acordarse del flag.

Editar `CFG`/`PROFILE` (config.py) y correr:
    make train-dataset ARGS="--levels 1,9,12"                    # perfil PROFILE (default)
    make train-dataset ARGS="--levels 12 --profile fast"         # un nivel puntual, perfil fast
    # o
    uv run python -m scripts.train_dataset --levels 1,9,12 --profile optimized

Qué niveles/grains se entrenan (sin --levels) se controla en config.ACTIVE_LEVEL_IDS
y `level.grains` (config/levels.py).

Niveles con `split_by` (10-12: un dataset por combinación dept/store, ver
config/levels.py) entrenan un modelo por combinación, iterando sobre los
parquets ya generados por build-datasets (ver main()).

Salidas:
    artifacts/results/{level}_sales_model_comparison.csv   comparación de modelos, con columna
                                                              `stage` (naive/statistical/bench/
                                                              post_feature_selection/final)
    artifacts/results/{level}_sales_series_metrics.csv     métricas por serie de esos modelos
    artifacts/results/{level}_sales_feature_selection.csv  historial de backward elimination
    artifacts/plots/{level}_sales_shap_*.png                explicación SHAP del ganador
    artifacts/optuna_study.db                               estudios Optuna (sqlite, uno por
                                                               level/grain/familia/etapa)
    artifacts/models/{level}_sales_artifact.pkl            artifact liviano (modelo + features +
                                                               métricas + winner_family + explainer
                                                               SHAP, sin datos -- ver
                                                               src.data.dataset.reconstruct_test_data())
"""
import argparse
import gc
from dataclasses import asdict, replace

from loguru import logger

import config
from src.logging_setup import configure_logging
from scripts.train_dataset.config import CFG, PROFILE
from scripts.train_dataset.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="Entrena modelos ML (bench + Optuna sobre el ganador) por nivel de agregación M5.")
    ap.add_argument("--levels", help="IDs separados por coma, p.ej. 1,9,12. "
                    "Default: config.ACTIVE_LEVEL_IDS.")
    ap.add_argument("--grain", default=None, choices=["daily", "weekly"],
                    help="Granularidad a entrenar. Default: todas las de level.grains "
                         "(hoy daily y weekly en todos los niveles, ver config/levels.py) -- "
                         "una corrida por grain, cada una con su propio dataset.")
    ap.add_argument("--profile", default=PROFILE,
                    choices=sorted(config.TRAINING_PROFILES),
                    help="Perfil de entrenamiento (ver config/training.py): fast (smoke-test, "
                         "sin stats/feature-selection/shap/optuna final), moderate (primera "
                         "versión en producción, sin las fases caras de interpretabilidad) u "
                         f"optimized (pipeline completo, para las métricas finales del TFM). "
                         f"Default: {PROFILE!r} (ver PROFILE en config.py).")
    args = ap.parse_args()

    level_ids = [int(x) for x in args.levels.split(",")] if args.levels else list(config.ACTIVE_LEVEL_IDS)
    profile = config.TRAINING_PROFILES[args.profile]
    profile_overrides = {k: v for k, v in asdict(profile).items() if k != "name"}
    logger.info("Perfil de entrenamiento: {}", profile.name)

    for level_id in level_ids:
        level = config.LEVELS_BY_ID[level_id]
        grains = [args.grain] if args.grain else list(level.grains)

        for grain in grains:
            if grain not in level.grains:
                logger.warning("Nivel {} ({}): no soporta grain {!r} (grains={}) -- se saltea.",
                                level_id, level.name, grain, level.grains)
                continue

            cfg = replace(CFG, level_id=level_id, grain=grain, **profile_overrides)

            if not level.split_by:
                try:
                    run_pipeline(cfg)
                except Exception:
                    logger.exception("Nivel {} grain {} falló, sigue con el resto.", level_id, grain)
                gc.collect()
                continue

            # split_by (10-12): un modelo por combinación (p.ej. dept_id x store_id en L12),
            # un parquet por combinación ya generado por build-datasets.
            combos = config.featured_level_combos(level, grain)
            if not combos:
                logger.warning("Nivel {} ({}) grain {}: no hay parquets de combinación en "
                                "data/datasets/{}/ -- correr build-datasets primero. Se saltea.",
                                level_id, level.name, grain, grain)
                continue
            logger.info("Nivel {} ({}) grain {}: {} combinaciones", level_id, level.name, grain, len(combos))
            for path in combos:
                try:
                    run_pipeline(cfg, dataset_path=path)
                except Exception:
                    logger.exception("Nivel {} combo {} grain {} falló, sigue con el resto.",
                                      level_id, path.stem, grain)
                gc.collect()


if __name__ == "__main__":
    configure_logging("train_dataset")
    main()
