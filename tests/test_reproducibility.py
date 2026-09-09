"""Harness de reproducibilidad (Fase 5 del plan de testing): confirma que las 5
familias de src/modeling/families.py son deterministas dado el mismo random_state
-- la garantía real detrás de `cfg.random_state` (scripts/train_dataset/config.py),
que hoy se confía mucho pero no se verifica en ningún test. Si alguien introduce
aleatoriedad no seedeada (p.ej. un sample() sin random_state, o un default de
librería que cambia entre versiones), este test debería detectarlo.

Marcado `slow` (entrena 5 modelos reales, aunque sobre datos sintéticos chicos) --
se excluye del run rápido por defecto, ver pyproject.toml y README."""
import numpy as np
import pandas as pd
import pytest

from src.modeling.families import ALL_FAMILIES, MODEL_FAMILIES


def _synthetic_frame(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    idx = np.arange(n)
    X = pd.DataFrame({
        "num1": np.sin(idx / 5.0),
        "num2": (idx % 17) / 17.0,
        "num3": np.cos(idx / 7.0) * 2,
        "cat1": pd.Categorical([f"c{i % 3}" for i in idx]),
    })
    y = X["num1"] * 3 + X["num2"] * 2 - X["num3"] + X["cat1"].cat.codes.astype(float) * 0.5
    return X, y.astype(float)


@pytest.mark.slow
@pytest.mark.parametrize("family_name", ALL_FAMILIES)
def test_family_fit_is_deterministic_given_a_fixed_seed(family_name):
    X, y = _synthetic_frame()
    split = 150
    X_train, X_valid = X.iloc[:split].reset_index(drop=True), X.iloc[split:].reset_index(drop=True)
    y_train, y_valid = y.iloc[:split].reset_index(drop=True), y.iloc[split:].reset_index(drop=True)

    family = MODEL_FAMILIES[family_name]
    categorical_features = ["cat1"]
    numerical_features = ["num1", "num2", "num3"]
    params = family.resolve_objective("regression_l2", 1.5)
    if family.n_estimators_param:
        params = {**params, family.n_estimators_param: 30}

    def _fit_and_predict():
        fitted = family.fit(
            X_train, y_train, X_valid, y_valid,
            categorical_features, numerical_features,
            params, early_stopping_rounds=None, random_state=42,
        )
        return fitted.predict(X_valid)

    preds1 = _fit_and_predict()
    preds2 = _fit_and_predict()
    assert np.array_equal(preds1, preds2), (
        f"{family_name}: dos fits con el mismo random_state dieron predicciones "
        "distintas -- alguna fuente de aleatoriedad no está seedeada."
    )
