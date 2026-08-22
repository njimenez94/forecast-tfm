"""Modelos de referencia para comparar antes de invertir en feature selection y
tuning (ver notebook 02_model, sección "Comparación de modelos").

`naive_last_value` y `seasonal_naive` no entrenan nada: son el piso mínimo que
cualquier modelo debe superar. Los `fit_*` entrenan con hiperparámetros por
defecto (sin tuning) sobre train, para comparar familias de modelo en
validación vía `evaluate_predictions` (`src.evaluation.metrics`) antes de
elegir una y recién ahí iterar (selección de features, Optuna, etc.).
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


def naive_last_value(train_df, valid_df, target_col, group_col="agg_id"):
    """Repite la última venta observada por serie (día anterior al inicio de validación)."""
    last_train_value = train_df.sort_values("date").groupby(group_col)[target_col].last()
    return valid_df[group_col].map(last_train_value).fillna(0.0).to_numpy()


def seasonal_naive(train_df, valid_df, target_col, group_col="agg_id", season_length=7):
    """Repite el valor observado el mismo día de la semana en la última semana de
    train (estacionalidad semanal, típica en ventas diarias de retail)."""
    last_season = (
        train_df.sort_values("date")
        .groupby(group_col)[target_col]
        .apply(lambda s: s.to_numpy()[-season_length:])
    )

    valid_sorted = valid_df.sort_values([group_col, "date"])
    step_in_season = valid_sorted.groupby(group_col).cumcount().to_numpy()

    y_pred_sorted = np.array([
        season[i % len(season)] if (season := last_season.get(gid)) is not None and len(season) > 0 else 0.0
        for gid, i in zip(valid_sorted[group_col], step_in_season)
    ])

    return (
        pd.Series(y_pred_sorted, index=valid_sorted.index)
        .reindex(valid_df.index)
        .to_numpy()
    )


def fit_lightgbm(X_train, y_train, categorical_features, random_state=42):
    model = LGBMRegressor(objective="rmse", random_state=random_state, verbosity=-1)
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


def fit_histgb(X_train, y_train, random_state=42):
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        categorical_features="from_dtype",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def fit_ridge(X_train, y_train, numerical_features, random_state=42):
    """Modelo lineal de referencia: solo usa las features numéricas (no maneja
    categóricas de alta cardinalidad)."""
    model = make_pipeline(SimpleImputer(strategy="median"), Ridge(random_state=random_state))
    model.fit(X_train[numerical_features], y_train)
    return model
