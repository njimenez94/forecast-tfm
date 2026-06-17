"""Hiperparámetros de LightGBM."""

SEED = 42

LGBM_PARAMS = {
    "objective": "regression_l2",
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