import lightgbm as lgb
import pandas as pd
from loguru import logger
from sklearn.inspection import permutation_importance

from src.evaluation.metrics import clip_closed_stores, compute_wrmsse


def compute_permutation_importance(model, X_valid, y_valid, features,
                                    sample_size: int = 15_000, n_repeats: int = 3,
                                    random_state: int = 42) -> pd.Series:
    """Permutation importance (RMSE) de `model` sobre una muestra de validación,
    para no pagar `n_repeats` reentrenamientos completos. Pensada como ranking de
    entrada de `backward_feature_selection` (de menor a mayor importancia)."""
    X_sample = X_valid.sample(n=min(sample_size, len(X_valid)), random_state=random_state)
    y_sample = y_valid.loc[X_sample.index]

    result = permutation_importance(
        model, X_sample, y_sample,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    return pd.Series(result.importances_mean, index=features).sort_values(ascending=False)


def backward_feature_selection(X_train, y_train, X_valid, train_df, valid_df,
                                features, categorical_features, importance_perm,
                                tolerance: float = 0.001, random_state: int = 42):
    """Elimina features de forma iterativa mientras el WRMSSE de validación no
    empeore más que `tolerance`. Parte del ranking de permutation importance (de
    menor a mayor) e intenta descartar cada feature reentrenando sin ella."""

    def train_and_eval(feats):
        cat_feats = [c for c in categorical_features if c in feats]
        m = lgb.LGBMRegressor(objective="rmse", random_state=random_state, verbosity=-1)
        m.fit(X_train[feats], y_train, categorical_feature=cat_feats)
        y_pred = clip_closed_stores(valid_df, m.predict(X_valid[feats]))
        return compute_wrmsse(train_df, valid_df, y_pred)

    current_features = list(features)
    best_wrmsse = train_and_eval(current_features)
    selection_log = [{"n_features": len(current_features), "wrmsse": best_wrmsse, "removed": None, "accepted": True}]
    logger.info(f"Baseline ({len(current_features)} features): WRMSSE={best_wrmsse:.4f}")

    improved = True
    while improved:
        improved = False
        ranking = importance_perm[current_features].sort_values().index.tolist()
        for feat in ranking:
            candidate_features = [f for f in current_features if f != feat]
            wrmsse_candidate = train_and_eval(candidate_features)
            accepted = wrmsse_candidate <= best_wrmsse + tolerance
            selection_log.append({"n_features": len(candidate_features), "wrmsse": wrmsse_candidate, "removed": feat, "accepted": accepted})

            if accepted:
                current_features = candidate_features
                best_wrmsse = wrmsse_candidate
                improved = True
                logger.warning(f"ELIMINADA: '{feat}' -> {len(current_features)} features | WRMSSE={wrmsse_candidate:.4f}")
            else:
                logger.success(f"mantiene: '{feat}' | WRMSSE empeoraría a {wrmsse_candidate:.4f}")

    logger.info(f"Seleccionadas {len(current_features)}/{len(features)} features | WRMSSE final: {best_wrmsse:.4f}")
    return current_features, best_wrmsse, pd.DataFrame(selection_log)
