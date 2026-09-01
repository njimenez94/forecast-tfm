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
                                features, categorical_features, importance_perm, fit_predict_fn,
                                objective: str = "rmse", tweedie_variance_power: float = 1.5,
                                tolerance: float = 0.001, random_state: int = 42,
                                max_batch_size: int = 8):
    """Greedy backward feature selection guiado por permutation importance.

    Parte del set completo de `features` e intenta eliminarlas en bloques de
    hasta `max_batch_size`, en orden ascendente de `importance_perm` (las
    menos importantes primero). Cada eliminación (de bloque o individual) se
    acepta si `objective_metric` (rmse o deviance de Tweedie, según
    `objective`) en validación no empeora más que `tolerance` respecto al
    mejor valor actual; si se acepta, las features quedan fuera y el ranking
    se recalcula sobre el subconjunto restante (importancias relativas
    cambian tras cada eliminación). El proceso itera hasta que una pasada
    completa no logra eliminar ninguna feature. El WRMSSE también se calcula
    en cada paso, pero solo para informar/graficar -- quien decide es
    `objective_metric`, coherente con la pérdida real del modelo.

    Es "greedy" porque acepta la primera eliminación válida de cada
    iteración en vez de evaluar todas las combinaciones posibles y elegir
    la óptima (exhaustive search) — más rápido, no garantiza el mínimo
    global de la métrica, pero es la aproximación estándar quue se usa en la
    práctica cuando el reentrenamiento es costoso.

    Batching (`max_batch_size`): con cientos de features candidatas, probar
    la eliminación de a una es O(n_features^2) reentrenamientos de LightGBM.
    Cada intento prueba remover un bloque de hasta `max_batch_size` features
    (las siguientes en el ranking) de una sola vez; si se rechaza, cae a
    probar esas mismas features una por una antes de darlas por buenas, así
    que nunca se pierde una eliminación individual válida solo por venir en
    un bloque rechazado (el costo extra de un bloque rechazado es un único
    reentrenamiento de más -- el del bloque -- sobre el baseline uno a uno).
    `max_batch_size=1` reproduce el algoritmo original exactamente.

    OJO -- esto NO garantiza el mismo resultado final que correr todo uno a
    uno: probar un bloque completo es una condición más fuerte que probar
    sus features por separado (aceptar el bloque implica que remover TODAS
    esas features a la vez no empeora la métrica), lo que puede detectar
    grupos de features conjuntamente redundantes/correlacionadas que el
    modo uno-a-uno deja adentro porque, individualmente, remover cada una
    por separado sí empeora un poco (el modelo se apoya circunstancialmente
    en las que quedan del grupo). En la práctica esto suele traducirse en
    una selección final más chica (más agresiva podando redundancia), no en
    perder features que sí aportan -- pero al ser greedy, el orden de las
    pruebas importa y el óptimo global no está garantizado en ninguno de
    los dos modos (ver párrafo anterior). Para reproducir bit a bit una
    corrida anterior sin batching, usar `max_batch_size=1`.

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
        fit_predict_fn: callable `(feats, cat_feats) -> y_pred_valid` que entrena el
            modelo candidato (familia y sus hiperparámetros ya fijos -- ver
            `src.modeling.families` y `scripts/train_dataset.py:select_features`)
            sobre `X_train[feats]`/`y_train` y devuelve sus predicciones sobre
            `X_valid[feats]`. Reentrenado en cada intento de eliminación.
        objective: objective del modelo ("rmse" o "tweedie") usado para decidir
            la eliminación (vía `objective_metric`); coherente con el objective
            real con el que `fit_predict_fn` entrena cada candidato.
        tweedie_variance_power: power de Tweedie, solo aplica si
            `objective == "tweedie"`.
        tolerance: máximo empeoramiento RELATIVO de `objective_metric` (fracción
            de `best_score`, no valor absoluto) aceptable para eliminar
            features -- p.ej. 0.001 = hasta 0.1% peor. Relativo porque la
            escala de `objective_metric` varía mucho entre niveles (L1 total
            vs. L12 item_store), así que un mismo `tolerance` absoluto no
            significa lo mismo en todos los niveles; en cambio, un umbral
            relativo sí es comparable. A mayor tolerance, selección más
            agresiva y (con `max_batch_size>1`) más bloques se aceptan
            enteros en vez de caer al fallback uno a uno, porque con
            tolerance=0 el más mínimo ruido numérico entre entrenamientos ya
            alcanza para rechazar cualquier bloque de más de una feature.
        random_state: semilla para reproducibilidad del LGBMRegressor.
        max_batch_size: máximo de features candidatas a remover juntas en un
            solo reentrenamiento (ver "Batching" arriba). 1 = sin batching.

    Returns:
        selected_features: lista final de features tras la selección.
        best_score: `objective_metric` de validación del modelo final (la que decidió la selección).
        best_wrmsse: WRMSSE de validación del modelo final (informativo).
        selection_log: DataFrame con el historial de cada intento
            (n_features, score, wrmsse, features removidas -- una o varias
            separadas por ";" --, cuántas se intentó remover, si fue aceptado).
    """

    def train_and_eval(feats):
        cat_feats = [c for c in categorical_features if c in feats]
        y_pred = clip_closed_stores(valid_df, fit_predict_fn(feats, cat_feats))
        score = objective_metric(y_valid, y_pred, objective, tweedie_variance_power)
        wrmsse = compute_wrmsse(train_df, valid_df, y_pred)
        return score, wrmsse

    current_features = list(features)
    best_score, best_wrmsse = train_and_eval(current_features)
    selection_log = [{"n_features": len(current_features), "score": best_score, "wrmsse": best_wrmsse, "removed": None, "n_removed": 0, "accepted": True}]
    logger.info(f"Baseline ({len(current_features)} features): {objective}={best_score:.4f} | WRMSSE={best_wrmsse:.4f} (informativo)")

    def try_remove(batch: list[str]) -> bool:
        """Intenta remover `batch` (una o varias features) de `current_features`.
        Actualiza current_features/best_score/best_wrmsse/improved (nonlocal) y
        loguea el intento si se acepta. Devuelve si se aceptó, para que el
        llamador decida si hace falta el fallback uno-a-uno (ver batching en
        el docstring)."""
        nonlocal current_features, best_score, best_wrmsse, improved
        candidate_features = [f for f in current_features if f not in batch]
        score_candidate, wrmsse_candidate = train_and_eval(candidate_features)
        accepted = score_candidate <= best_score * (1 + tolerance)
        selection_log.append({
            "n_features": len(candidate_features), "score": score_candidate,
            "wrmsse": wrmsse_candidate, "removed": ";".join(batch),
            "n_removed": len(batch), "accepted": accepted,
        })
        if accepted:
            current_features = candidate_features
            best_score = score_candidate
            best_wrmsse = wrmsse_candidate
            improved = True
            label = f"{len(batch)} features {batch}" if len(batch) > 1 else f"'{batch[0]}'"
            logger.warning(f"ELIMINADA(S) {label} -> {len(current_features)} features | {objective}={score_candidate:.4f} | WRMSSE={wrmsse_candidate:.4f}")
        return accepted

    improved = True
    while improved:
        improved = False
        ranking = importance_perm[current_features].sort_values().index.tolist()
        i = 0
        while i < len(ranking):
            # El batch nunca debe cubrir TODAS las features que quedan --
            # dejaría 0 features (LightGBM no puede entrenar sin ninguna).
            size = min(max_batch_size, len(ranking) - i)
            if size >= len(current_features):
                size = len(current_features) - 1
            if size <= 0:
                break
            batch = ranking[i:i + size]
            if len(batch) == 1:
                try_remove(batch)
            elif not try_remove(batch):
                # Bloque rechazado: no asumimos que ninguna sea removible,
                # se prueban una por una (igual que el algoritmo original)
                # para no perder eliminaciones válidas dentro del bloque.
                for feat in batch:
                    try_remove([feat])
            i += len(batch)

    logger.info(f"Seleccionadas {len(current_features)}/{len(features)} features | {objective} final: {best_score:.4f} | WRMSSE final: {best_wrmsse:.4f}")
    return current_features, best_score, best_wrmsse, pd.DataFrame(selection_log)
