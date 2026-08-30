"""Wrappers de gradient boosting (LightGBM/XGBoost/CatBoost/HistGB): entrenan con
hiperparámetros por defecto (sin tuning) sobre la matriz de features
(X_train/y_train)."""
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor


def fit_lightgbm(X_train, y_train, categorical_features, objective="rmse",
                  tweedie_variance_power=1.5, random_state=42):
    params = {"objective": objective, "random_state": random_state, "verbosity": -1}
    if objective == "tweedie":
        params["tweedie_variance_power"] = tweedie_variance_power
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, categorical_feature=categorical_features)
    return model


def fit_xgboost(X_train, y_train, random_state=42):
    """Requiere que las columnas categóricas ya sean dtype 'category' (usa
    `enable_categorical=True`, sin necesidad de encodear a mano)."""
    model = XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist",
        enable_categorical=True,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def catboost_features(X, categorical_features):
    """CatBoost no soporta el dtype category de pandas con nulos: convierte las
    categóricas a string, con los NaN reemplazados por 'missing'. Aplicar por
    igual a train (antes de `fit_catboost`) y a valid/test (antes de `predict`)."""
    X_cb = X.copy()
    for col in categorical_features:
        X_cb[col] = X_cb[col].astype(str).fillna("missing")
    return X_cb


def fit_catboost(X_train, y_train, categorical_features, random_state=42):
    X_train_cb = catboost_features(X_train, categorical_features)
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_state=random_state,
        cat_features=categorical_features,
        verbose=False,
    )
    model.fit(X_train_cb, y_train)
    return model


def histgb_features(X, high_cardinality_features):
    """Convierte a códigos enteros las categóricas de alta cardinalidad que
    HistGB no puede tratar como categóricas nativas (ver `fit_histgb`), para
    que no lleguen como strings a `check_array`. Aplicar por igual a train
    (antes de `fit_histgb`) y a valid/test (antes de `predict`)."""
    X_hgb = X.copy()
    for col in high_cardinality_features:
        X_hgb[col] = X_hgb[col].cat.codes
    return X_hgb


def fit_histgb(X_train, y_train, random_state=42, max_bins=255):
    """HistGradientBoostingRegressor no soporta categóricas con cardinalidad >
    max_bins (255 por defecto): a diferencia de LightGBM/XGBoost/CatBoost, no
    puede tratar `item_id` (niveles 10-12, ~3049 valores) como categórica
    nativa. Se detectan las columnas dtype 'category' y solo se pasan como
    categóricas las que caben dentro del límite; el resto (p. ej. item_id) se
    convierte a códigos enteros con `histgb_features` para que HistGB la
    trate como numérica, evitando el ValueError."""
    categorical_features = [
        col
        for col in X_train.columns
        if isinstance(X_train[col].dtype, pd.CategoricalDtype)
        and len(X_train[col].cat.categories) <= max_bins
    ]
    high_cardinality_features = [
        col
        for col in X_train.columns
        if isinstance(X_train[col].dtype, pd.CategoricalDtype)
        and len(X_train[col].cat.categories) > max_bins
    ]
    X_train_hgb = histgb_features(X_train, high_cardinality_features)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        categorical_features=categorical_features,
        max_bins=max_bins,
        random_state=random_state,
    )
    model.fit(X_train_hgb, y_train)
    model.high_cardinality_features_ = high_cardinality_features
    return model
