"""Hiperparámetros de LightGBM."""

SEED = 42

LGBM_PARAMS = {
    "objective": "tweedie",
    # Encontrado por Optuna sobre WRMSSE/WAPE en notebooks/02_model.ipynb (celda de
    # búsqueda de hiperparámetros): tweedie maneja mejor la intermitencia
    # (abundancia de ceros) que regression_l2, sobre todo en periodos post-evento
    # (ver days_to_closure/days_to_thanksgiving/days_to_newyear).
    "tweedie_variance_power": 1.3124217733388135,
    "metric": "mae",
    "learning_rate": 0.08,
    "n_estimators": 200,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "max_bin": 63,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
}