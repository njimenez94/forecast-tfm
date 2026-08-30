import lightgbm as lgb
import pandas as pd
from loguru import logger
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from src.evaluation.objectives import objective_metric
from src.evaluation.scaled import clip_closed_stores, compute_wrmsse


def compute_permutation_importance(model, X_valid, y_valid, features, objective: str = "rmse",
                                    tweedie_variance_power: float = 1.5, sample_size: int = 15_000,
                                    n_repeats: int = 3, random_state: int = 42) -> pd.Series:
    """Permutation importance de `model` sobre una muestra de validación, para no
    pagar `n_repeats` reentrenamientos completos. Pensada como ranking de entrada
    de `backward_feature_selection` (de menor a mayor importancia).

    Usa `objective` (rmse o tweedie, ver config.lgbm_params) como criterio de
    scoring, coherente con la pérdida real del modelo en vez de forzar siempre RMSE."""
    X_sample = X_valid.sample(n=min(sample_size, len(X_valid)), random_state=random_state)
    y_sample = y_valid.loc[X_sample.index]

    scorer = make_scorer(objective_metric, objective=objective,
                          tweedie_variance_power=tweedie_variance_power, greater_is_better=False)
    result = permutation_importance(
        model, X_sample, y_sample,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    return pd.Series(result.importances_mean, index=features).sort_values(ascending=False)


def backward_feature_selection(X_train, y_train, X_valid, y_valid, train_df, valid_df,
                                features, categorical_features, importance_perm,
                                objective: str = "rmse", tweedie_variance_power: float = 1.5,
                                tolerance: float = 0.001, random_state: int = 42):
    """Greedy backward feature selection guiado por permutation importance.

    Parte del set completo de `features` e intenta eliminarlas una a una,
    en orden ascendente de `importance_perm` (las menos importantes primero).
    Cada eliminación se acepta si `objective_metric` (rmse o deviance de
    Tweedie, según `objective`) en validación no empeora más que `tolerance`
    respecto al mejor valor actual; si se acepta, la feature queda fuera y el
    ranking se recalcula sobre el subconjunto restante (importancias relativas
    cambian tras cada eliminación). El proceso itera hasta que una pasada
    completa no logra eliminar ninguna feature. El WRMSSE también se calcula
    en cada paso, pero solo para informar/graficar -- quien decide es
    `objective_metric`, coherente con la pérdida real del modelo.

    Es "greedy" porque acepta la primera eliminación válida de cada
    iteración en vez de evaluar todas las combinaciones posibles y elegir
    la óptima (exhaustive search) — más rápido, no garantiza el mínimo
    global de la métrica, pero es la aproximación estándar quue se usa en la
    práctica cuando el reentrenamiento es costoso.

    Args:
        X_train, y_train: features y target de entrenamiento.
        X_valid, y_valid: features y target de validación.
        train_df, valid_df: dataframes originales, requeridos por
            `clip_closed_stores` y `compute_wrmsse` (métrica jerárquica).
        features: lista inicial completa de features candidatas.
        categorical_features: subset de `features` a tratar como categóricas
            en LightGBM (se filtra dinámicamente en cada iteración).
        importance_perm: ranking de permutation importance (Series o dict
            indexado por nombre de feature) usado para decidir el orden de
            intento de eliminación.
        objective: objective de LightGBM ("rmse" o "tweedie", ver
            config.lgbm_params) usado tanto para entrenar cada candidato como
            para decidir la eliminación (vía `objective_metric`).
        tweedie_variance_power: power de Tweedie, solo aplica si
            `objective == "tweedie"`.
        tolerance: máximo empeoramiento de `objective_metric` aceptable para
            eliminar una feature. A mayor tolerance, selección más agresiva.
        random_state: semilla para reproducibilidad del LGBMRegressor.

    Returns:
        selected_features: lista final de features tras la selección.
        best_score: `objective_metric` de validación del modelo final (la que decidió la selección).
        best_wrmsse: WRMSSE de validación del modelo final (informativo).
        selection_log: DataFrame con el historial de cada intento
            (n_features, score, wrmsse, feature removida, si fue aceptada).
    """

    def train_and_eval(feats):
        cat_feats = [c for c in categorical_features if c in feats]
        params = {"objective": objective, "random_state": random_state, "verbosity": -1}
        if objective == "tweedie":
            params["tweedie_variance_power"] = tweedie_variance_power
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train[feats], y_train, categorical_feature=cat_feats)
        y_pred = clip_closed_stores(valid_df, m.predict(X_valid[feats]))
        score = objective_metric(y_valid, y_pred, objective, tweedie_variance_power)
        wrmsse = compute_wrmsse(train_df, valid_df, y_pred)
        return score, wrmsse

    current_features = list(features)
    best_score, best_wrmsse = train_and_eval(current_features)
    selection_log = [{"n_features": len(current_features), "score": best_score, "wrmsse": best_wrmsse, "removed": None, "accepted": True}]
    logger.info(f"Baseline ({len(current_features)} features): {objective}={best_score:.4f} | WRMSSE={best_wrmsse:.4f} (informativo)")

    improved = True
    while improved:
        improved = False
        ranking = importance_perm[current_features].sort_values().index.tolist()
        for feat in ranking:
            candidate_features = [f for f in current_features if f != feat]
            score_candidate, wrmsse_candidate = train_and_eval(candidate_features)
            accepted = score_candidate <= best_score + tolerance
            selection_log.append({"n_features": len(candidate_features), "score": score_candidate, "wrmsse": wrmsse_candidate, "removed": feat, "accepted": accepted})

            if accepted:
                current_features = candidate_features
                best_score = score_candidate
                best_wrmsse = wrmsse_candidate
                improved = True
                logger.warning(f"ELIMINADA: '{feat}' -> {len(current_features)} features | {objective}={score_candidate:.4f} | WRMSSE={wrmsse_candidate:.4f}")
            else:
                #logger.success(f"mantiene: '{feat}' | {objective} empeoraría a {score_candidate:.4f}")
                continue

    logger.info(f"Seleccionadas {len(current_features)}/{len(features)} features | {objective} final: {best_score:.4f} | WRMSSE final: {best_wrmsse:.4f}")
    return current_features, best_score, best_wrmsse, pd.DataFrame(selection_log)
