"""Hiperparámetros de LightGBM."""

SEED = 42

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
