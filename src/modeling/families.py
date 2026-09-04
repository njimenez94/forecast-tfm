"""Registro de familias de modelo ML (LightGBM/XGBoost/CatBoost/HistGB/Ridge): una
interfaz uniforme (fit con hiperparámetros arbitrarios + early stopping, predict,
espacio de búsqueda Optuna, importancia nativa) para que `scripts/train_dataset.py`
pueda tunear, comparar y promover a "ganador" cualquiera de las cinco sin
hardcodear LightGBM en cada fase.

Ridge es la única familia no tree-based (`is_tree_based=False`): participa del bench
de Optuna pero nunca puede ser el modelo ganador (no es compatible con
`shap.TreeExplainer`, ver `pick_winner` en train_dataset.py)."""
import functools
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin

from src.modeling.gradient_boosting import catboost_features, histgb_features


def _identity(X):
    return X


def _select_columns(X, columns):
    return X[columns]


class FittedModel(RegressorMixin, BaseEstimator):
    """Wrapper uniforme: `.predict(X)` aplica el preprocesamiento propio de la
    familia (catboost/histgb dtype wrangling, subset numérico de Ridge) antes de
    llamar al estimador nativo. `estimator`/`prepare` quedan expuestos por separado
    para SHAP (TreeExplainer necesita el estimador nativo + X ya preparada).

    Hereda de `RegressorMixin`/`BaseEstimator` (en vez de ser un dataclass simple)
    porque `sklearn.inspection.permutation_importance` (usada por
    `compute_permutation_importance`, ver `src/modeling/feature_selection.py`) valida
    vía `__sklearn_tags__`/`is_regressor` que el estimador venga de esa jerarquía --
    no llama a `.fit()` (definido acá como no-op, el estimador ya viene entrenado).
    """

    def __init__(self, estimator, prepare: Callable[[pd.DataFrame], pd.DataFrame],
                 best_iteration: int | None = None):
        self.estimator = estimator
        self.prepare = prepare
        self.best_iteration = best_iteration

    def predict(self, X):
        return self.estimator.predict(self.prepare(X))

    def fit(self, X, y=None):
        return self


# ============================== LightGBM ==============================

def _lgbm_objective(objective: str, tweedie_variance_power: float) -> dict:
    kwargs = {"objective": objective}
    if objective == "tweedie":
        kwargs["tweedie_variance_power"] = tweedie_variance_power
    return kwargs


def _lgbm_space(trial, cfg) -> dict:
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        n_estimators=cfg.optuna_n_estimators,
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        max_depth=trial.suggest_int("max_depth", 5, 10),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5, log=True),
    )
    if cfg.objective == "tweedie":
        params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return params


def _fit_lightgbm(X_train, y_train, X_valid, y_valid, categorical_features, numerical_features,
                   params, early_stopping_rounds, random_state) -> FittedModel:
    import lightgbm as lgb

    model_params = {**params, "random_state": random_state, "n_jobs": -1, "verbosity": -1}
    model = lgb.LGBMRegressor(**model_params)
    eval_metric = "tweedie" if params.get("objective") == "tweedie" else "rmse"
    callbacks = [lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)] \
        if early_stopping_rounds else []
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=eval_metric,
              categorical_feature=categorical_features, callbacks=callbacks)
    return FittedModel(model, _identity, getattr(model, "best_iteration_", None))


def _lgbm_gain_importance(fitted: FittedModel, features: list[str]) -> pd.Series:
    return pd.Series(
        fitted.estimator.booster_.feature_importance(importance_type="gain"), index=features,
    ).sort_values(ascending=False)


# ============================== XGBoost ==============================

def _xgb_objective(objective: str, tweedie_variance_power: float) -> dict:
    if objective == "tweedie":
        return {"objective": "reg:tweedie", "tweedie_variance_power": tweedie_variance_power}
    return {"objective": "reg:absoluteerror"}


def _xgb_space(trial, cfg) -> dict:
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        n_estimators=cfg.optuna_n_estimators,
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5, log=True),
    )
    if cfg.objective == "tweedie":
        params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
    return params


def _fit_xgboost(X_train, y_train, X_valid, y_valid, categorical_features, numerical_features,
                  params, early_stopping_rounds, random_state) -> FittedModel:
    from xgboost import XGBRegressor

    model_params = {
        **params, "tree_method": "hist", "enable_categorical": True, "random_state": random_state,
    }
    if early_stopping_rounds:
        model_params["early_stopping_rounds"] = early_stopping_rounds
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return FittedModel(model, _identity, getattr(model, "best_iteration", None))


def _xgb_gain_importance(fitted: FittedModel, features: list[str]) -> pd.Series:
    score = fitted.estimator.get_booster().get_score(importance_type="gain")
    return pd.Series(score).reindex(features).fillna(0.0).sort_values(ascending=False)


# ============================== CatBoost ==============================

def _catboost_objective(objective: str, tweedie_variance_power: float) -> dict:
    if objective == "tweedie":
        return {"loss_function": f"Tweedie:variance_power={tweedie_variance_power}"}
    return {"loss_function": "RMSE"}


def _catboost_space(trial, cfg) -> dict:
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        n_estimators=cfg.optuna_n_estimators,
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
    )


def _fit_catboost(X_train, y_train, X_valid, y_valid, categorical_features, numerical_features,
                   params, early_stopping_rounds, random_state) -> FittedModel:
    from catboost import CatBoostRegressor

    prepare = functools.partial(catboost_features, categorical_features=categorical_features)
    model_params = {
        **params, "random_state": random_state, "cat_features": categorical_features,
        "verbose": False, "allow_writing_files": False,
    }
    model = CatBoostRegressor(**model_params)
    fit_kwargs = {"eval_set": (prepare(X_valid), y_valid)}
    if early_stopping_rounds:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    model.fit(prepare(X_train), y_train, **fit_kwargs)
    return FittedModel(model, prepare, model.get_best_iteration())


def _catboost_gain_importance(fitted: FittedModel, features: list[str]) -> pd.Series:
    return pd.Series(fitted.estimator.get_feature_importance(), index=features).sort_values(ascending=False)


# ============================== HistGB ==============================

def _histgb_objective(objective: str, tweedie_variance_power: float) -> dict:
    if objective == "tweedie":
        logger.warning(
            "HistGradientBoostingRegressor no soporta Tweedie nativo -- "
            "usando loss='poisson' como aproximación para el bench/tuning de esta familia.",
        )
        return {"loss": "poisson"}
    return {"loss": "absolute_error"}


def _histgb_space(trial, cfg) -> dict:
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        max_iter=cfg.optuna_n_estimators,
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255),
        max_depth=trial.suggest_int("max_depth", 3, 15),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 200),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 5, log=True),
    )


def _fit_histgb(X_train, y_train, X_valid, y_valid, categorical_features, numerical_features,
                params, early_stopping_rounds, random_state, max_bins=255) -> FittedModel:
    from sklearn.ensemble import HistGradientBoostingRegressor

    cat_feats = [
        col for col in X_train.columns
        if isinstance(X_train[col].dtype, pd.CategoricalDtype) and len(X_train[col].cat.categories) <= max_bins
    ]
    high_card = [
        col for col in X_train.columns
        if isinstance(X_train[col].dtype, pd.CategoricalDtype) and len(X_train[col].cat.categories) > max_bins
    ]
    prepare = functools.partial(histgb_features, high_cardinality_features=high_card)

    model_params = {**params, "categorical_features": cat_feats, "max_bins": max_bins, "random_state": random_state}
    if early_stopping_rounds:
        model_params.setdefault("early_stopping", True)
        model_params.setdefault("n_iter_no_change", early_stopping_rounds)
    else:
        # sklearn default es "auto" (early stopping implícito si hay >10k filas) --
        # lo desactivamos acá para que el refit final respete `max_iter` tal cual se
        # lo pasaron (mismo contrato que lightgbm/xgboost/catboost sin
        # early_stopping_rounds, ver fit_final_model).
        model_params.setdefault("early_stopping", False)
    model = HistGradientBoostingRegressor(**model_params)
    model.fit(prepare(X_train), y_train)
    return FittedModel(model, prepare, getattr(model, "n_iter_", None))


def _histgb_gain_importance(fitted: FittedModel, features: list[str]) -> None:
    return None


# ============================== Ridge ==============================

def _ridge_objective(objective: str, tweedie_variance_power: float) -> dict:
    return {}


def _ridge_space(trial, cfg) -> dict:
    return dict(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))


def _fit_ridge(X_train, y_train, X_valid, y_valid, categorical_features, numerical_features,
               params, early_stopping_rounds, random_state) -> FittedModel:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    prepare = functools.partial(_select_columns, columns=numerical_features)
    model = make_pipeline(SimpleImputer(strategy="median"), Ridge(random_state=random_state, **params))
    model.fit(prepare(X_train), y_train)
    return FittedModel(model, prepare, None)


def _ridge_gain_importance(fitted: FittedModel, features: list[str]) -> None:
    return None


# ============================== Registro ==============================

@dataclass
class ModelFamily:
    name: str
    is_tree_based: bool
    n_estimators_param: str | None
    resolve_objective: Callable[[str, float], dict]
    optuna_param_space: Callable
    fit: Callable[..., FittedModel]
    gain_importance: Callable[[FittedModel, list], "pd.Series | None"]


MODEL_FAMILIES: dict[str, ModelFamily] = {
    "lightgbm": ModelFamily(
        "lightgbm", True, "n_estimators", _lgbm_objective, _lgbm_space, _fit_lightgbm, _lgbm_gain_importance,
    ),
    "xgboost": ModelFamily(
        "xgboost", True, "n_estimators", _xgb_objective, _xgb_space, _fit_xgboost, _xgb_gain_importance,
    ),
    "catboost": ModelFamily(
        "catboost", True, "n_estimators", _catboost_objective, _catboost_space, _fit_catboost,
        _catboost_gain_importance,
    ),
    "histgb": ModelFamily(
        "histgb", True, "max_iter", _histgb_objective, _histgb_space, _fit_histgb, _histgb_gain_importance,
    ),
    "ridge": ModelFamily(
        "ridge", False, None, _ridge_objective, _ridge_space, _fit_ridge, _ridge_gain_importance,
    ),
}

TREE_FAMILIES = ("lightgbm", "xgboost", "catboost", "histgb")
ALL_FAMILIES = TREE_FAMILIES + ("ridge",)
