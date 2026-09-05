"""Hiperparámetros de LightGBM y qué targets entrenar por nivel."""

from config.features import CUM_EVAL_HORIZONS

SEED = 42

# Qué targets (objetivos) entrena `make train-dataset` por nivel y granularidad --
# fuente única de verdad para el default de scripts/train_dataset.py:main() (antes
# hardcodeado ahí como "todos los niveles activos x todos los cumN de
# CUM_EVAL_HORIZONS" por igual, sin poder ajustar por nivel ni por grain).
# Se indexa por grain (no solo por nivel) porque CUM_HORIZONS (config/features.py)
# define horizontes cumN distintos por granularidad (daily vs weekly) -- un nivel
# que corriera en "weekly" no puede usar la misma lista de cumN que uno "daily".
# Hoy todos los niveles (ver config/levels.py) son "daily", pero la forma queda
# lista para cuando eso cambie.
#
# --levels/--target en la CLI de train_dataset.py siguen pisando esto para
# corridas puntuales (p.ej. `make train-dataset ARGS="--levels 12 --target cum28"`).
# Editar acá para activar/desactivar targets puntuales por nivel -- p.ej. sacar los
# cumN más caros en los niveles item-level (10-12) sin tocar el resto.
TRAIN_TARGETS_BY_LEVEL: dict[int, dict[str, list[str]]] = {
    level_id: {"daily": ["sales", *(f"cum{n}" for n in CUM_EVAL_HORIZONS)]}
    for level_id in range(1, 13)
}

# Niveles con series muy intermitentes (item-level, alta cardinalidad, muchos
# ceros): "tweedie" maneja mejor esa abundancia de ceros que regression_l2 (el
# objective usado en el resto de los niveles, más agregados).
TWEEDIE_LEVELS = {11, 12}
TWEEDIE_VARIANCE_POWER = 1.5

# Fallback genérico (no tuneado para ningún nivel en particular) para cuando se
# entrena sin pasar por Optuna. El objective se decide por nivel, ver lgbm_params().
LGBM_PARAMS = {
    "metric": "mae",
    "learning_rate": 0.05,
    "n_estimators": 200,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
}


def lgbm_params(level_id: int) -> dict:
    """LGBM_PARAMS + objective según el nivel: tweedie para TWEEDIE_LEVELS
    (series intermitentes), regression_l2 para el resto."""
    params = dict(LGBM_PARAMS)
    if level_id in TWEEDIE_LEVELS:
        params["objective"] = "tweedie"
        params["tweedie_variance_power"] = TWEEDIE_VARIANCE_POWER
    else:
        params["objective"] = "regression_l2"
    return params
