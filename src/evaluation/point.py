"""Métricas puntuales de error, sin escalar por serie ni ponderar."""
import numpy as np


def wape(y_true, y_pred):
    """WAPE: error absoluto total como fracción de las ventas totales."""
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def bias(y_true, y_pred):
    """Bias del forecast; positivo = sobreestima, normalizado por ventas."""
    return (y_pred - y_true).sum() / np.abs(y_true).sum()


def smape(y_true, y_pred):
    """SMAPE: error porcentual simétrico promedio (con epsilon)."""
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return np.mean(2 * np.abs(y_pred - y_true) / denom)


def mae(y_true, y_pred):
    """MAE: error absoluto medio."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """RMSE: raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmsle(y_true, y_pred):
    """RMSLE: raíz del error cuadrático medio logarítmico. Penaliza más la
    subestimación que la sobreestimación y amortigua outliers en ventas altas
    (útil con series con muchos ceros/picos, como M5). Clipa negativos porque
    log1p no está definido para y < -1."""
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def tracking_signal(y_true, y_pred):
    """Tracking signal: error acumulado normalizado por el MAD (Brown, 1959).
    Estándar en demand forecasting/inventory management para detectar sesgo
    sistemático (no solo magnitud del error); valores fuera de ±4~6 se
    consideran señal de que el forecast está sistemáticamente sesgado."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    mad = np.mean(np.abs(error))
    if mad == 0:
        return 0.0
    return float(np.sum(error) / mad)
