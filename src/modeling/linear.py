"""Modelo lineal de referencia."""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


def fit_ridge(X_train, y_train, numerical_features, random_state=42):
    """Modelo lineal de referencia: solo usa las features numéricas (no maneja
    categóricas de alta cardinalidad)."""
    model = make_pipeline(SimpleImputer(strategy="median"), Ridge(random_state=random_state))
    model.fit(X_train[numerical_features], y_train)
    return model
