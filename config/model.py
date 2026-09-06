"""Hiperparámetros de LightGBM y qué targets entrenar por nivel."""

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
# Por decisión explícita para esta entrega del TFM: solo "sales" por ahora, los
# targets acumulados (cumN) quedan fuera de esta pasada para no complicar el
# alcance -- línea futura pendiente, no una limitación técnica del pipeline
# (que ya soporta cumN, ver config.CUM_EVAL_HORIZONS). Para reactivarlos en un
# nivel puntual: {"daily": ["sales", *(f"cum{n}" for n in config.features.CUM_EVAL_HORIZONS)]}
# "sales" corre en ambos grains (daily y weekly, ver config.levels.LEVELS) por igual.
#
# --levels/--target en la CLI de train_dataset.py siguen pisando esto para
# corridas puntuales (p.ej. `make train-dataset ARGS="--levels 12 --target cum28"`).
TRAIN_TARGETS_BY_LEVEL: dict[int, dict[str, list[str]]] = {
    level_id: {"daily": ["sales"], "weekly": ["sales"]}
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
