import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from loguru import logger

import config
from src.data.temporal_split import rolling_cv_folds
from src.evaluation import build_predictions_report, objective_metric
from src.evaluation.scaled import clip_closed_stores
from src.modeling import MODEL_FAMILIES, backward_feature_selection, compute_permutation_importance
from scripts.train_dataset.config import Config


def tune_optuna_family(state: SimpleNamespace, cfg: Config, family_name: str,
                        n_trials: int, timeout: int, study_name: str) -> dict:
    """Ronda de Optuna para una familia de modelo puntual: tunea `family.optuna_param_space`
    contra `objective_metric` (rmse/tweedie deviance, coherente con `cfg.objective` --
    el WRMSSE se guarda como user_attr solo para informar, igual que antes). Usada por
    `tune_optuna` (única ronda de Optuna del pipeline, sobre el ganador, post-feature-
    selection) -- el bench (benchmarking.run_bench_ml) no pasa por acá, usa
    hiperparámetros default.

    Devuelve los hiperparámetros combinados (`objective`/`n_estimators` fijos de la
    familia + los tuneados por Optuna) listos para pasar directo a `family.fit(...)` --
    a diferencia de `study.best_trial.params` a secas (que solo trae lo sugerido vía
    `trial.suggest_*`), así el caller no tiene que reconstruir el resto del dict.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    family = MODEL_FAMILIES[family_name]
    fixed_params = family.resolve_objective(cfg.objective, cfg.tweedie_variance_power)
    if family.n_estimators_param:
        fixed_params = {**fixed_params, family.n_estimators_param: cfg.optuna_n_estimators}

    def objective(trial):
        params = {**fixed_params, **family.optuna_param_space(trial, cfg)}
        fitted = family.fit(
            state.X_train, state.y_train, state.X_valid, state.y_valid,
            state.categorical_features, state.numerical_features,
            params, early_stopping_rounds=30, random_state=cfg.random_state,
        )
        y_pred = clip_closed_stores(state.valid, fitted.predict(state.X_valid))
        score = objective_metric(state.y_valid, y_pred, cfg.objective, cfg.tweedie_variance_power)
        _, wrmsse_val, _ = state.wrmsse_metric(state.y_valid, y_pred)
        trial.set_user_attr("wrmsse", wrmsse_val)
        return score

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{config.ARTIFACTS_DIR / 'optuna_study.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
    )
    logger.info("[{}] Optuna arrancando: hasta {} trials (timeout {}s) -- study {!r}",
                family_name, n_trials, timeout, study_name)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # user_attrs["wrmsse"] puede faltar en trials de corridas viejas del mismo
    # study persistido en sqlite (load_if_exists=True mezcla historial entre
    # ejecuciones) -- es solo informativo, no participa en qué hiperparámetros
    # elige Optuna (eso sale de study.best_trial.params).
    best_wrmsse = study.best_trial.user_attrs.get("wrmsse")
    wrmsse_str = f"{best_wrmsse:.4f}" if best_wrmsse is not None else "N/A (trial de una corrida anterior sin este dato)"
    logger.info(
        "[{}] Optuna terminó -- elige hiperparámetros por {} (no por WRMSSE): mejor {} = {:.4f} "
        "| WRMSSE de ESE trial: {} (solo referencia, puede no ser el mínimo posible; el WRMSSE "
        "definitivo se reporta abajo, tras reentrenar con estos hiperparámetros)",
        family_name, cfg.objective, cfg.objective, study.best_trial.value, wrmsse_str,
    )
    logger.info("[{}] Mejores hiperparámetros: {}", family_name, study.best_trial.params)

    return {**fixed_params, **study.best_trial.params}


def select_features(state: SimpleNamespace, cfg: Config) -> None:
    family = MODEL_FAMILIES[state.winner_family]
    fitted = state.bench_models[state.winner_family]
    params = state.bench_params[state.winner_family]

    importance_gain = family.gain_importance(fitted, state.features)
    if importance_gain is not None:
        logger.info("Top 10 gain importance ({}):\n{}", state.winner_family, importance_gain.head(10))

    importance_perm = compute_permutation_importance(
        fitted, state.X_valid, state.y_valid, state.features,
        sample_size=cfg.permutation_sample_size, n_repeats=cfg.permutation_n_repeats,
        random_state=cfg.random_state, objective=cfg.objective, tweedie_variance_power=cfg.tweedie_variance_power,
    )
    logger.info("Top 10 permutation importance ({}):\n{}", state.winner_family, importance_perm.head(10))

    def fit_predict_fn(feats, cat_feats):
        num_feats = [f for f in state.numerical_features if f in feats]
        fitted_candidate = family.fit(
            state.X_train[feats], state.y_train, state.X_valid[feats], state.y_valid,
            cat_feats, num_feats, params, early_stopping_rounds=30, random_state=cfg.random_state,
        )
        return fitted_candidate.predict(state.X_valid[feats])

    selected_features, final_score, final_wrmsse, selection_log = backward_feature_selection(
        state.X_train, state.y_train, state.X_valid, state.y_valid, state.train, state.valid,
        state.features, state.categorical_features, importance_perm, fit_predict_fn,
        objective=cfg.objective, tweedie_variance_power=cfg.tweedie_variance_power,
        tolerance=cfg.backward_tolerance, random_state=cfg.random_state,
        max_batch_size=cfg.feature_selection_batch_size,
    )

    dropped = sorted(set(state.features) - set(selected_features))
    logger.info("Features descartadas ({}): {}", len(dropped), dropped)

    state.features = selected_features
    state.categorical_features = [c for c in state.categorical_features if c in state.features]
    state.numerical_features = [c for c in state.numerical_features if c in state.features]
    state.X_train = state.X_train[state.features]
    state.X_valid = state.X_valid[state.features]
    state.X_test = state.X_test[state.features]

    out_dir = config.ARTIFACTS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_log.to_csv(out_dir / f"{state.level_str}_{state.target}_feature_selection.csv", index=False)

    logger.info("Features finales: {} ({} categóricas) | {}={:.4f} | WRMSSE={:.4f} (informativo)",
                len(state.features), len(state.categorical_features), cfg.objective, final_score, final_wrmsse)


def run_shap(state: SimpleNamespace, cfg: Config, evaluate_model) -> None:
    import matplotlib.pyplot as plt
    import shap

    family = MODEL_FAMILIES[state.winner_family]
    params = state.bench_params[state.winner_family]

    t0 = time.perf_counter()
    fitted = family.fit(
        state.X_train, state.y_train, state.X_valid, state.y_valid,
        state.categorical_features, state.numerical_features,
        params, early_stopping_rounds=100, random_state=cfg.random_state,
    )
    fit_time = time.perf_counter() - t0
    evaluate_model(f"{state.winner_family} (bench, selected features)", fitted.predict(state.X_valid),
                    fit_time=fit_time, category="ML", stage="post_feature_selection")

    explainer = shap.TreeExplainer(fitted.estimator)
    X_shap_raw = state.X_test.sample(n=min(cfg.shap_sample_size, len(state.X_test)), random_state=cfg.random_state)
    X_shap = fitted.prepare(X_shap_raw)
    shap_values = explainer.shap_values(X_shap)

    plot_dir = config.ARTIFACTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = plot_dir / f"{state.level_str}_{state.target}_shap"

    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{prefix}_bar.png", dpi=120)
    plt.close()

    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(f"{prefix}_beeswarm.png", dpi=120)
    plt.close()

    top_feature = X_shap.columns[np.argsort(-np.abs(shap_values).mean(axis=0))[0]]
    shap.dependence_plot(top_feature, shap_values, X_shap, interaction_index=None, show=False)
    plt.tight_layout()
    plt.savefig(f"{prefix}_dependence_{top_feature}.png", dpi=120)
    plt.close()

    logger.info("SHAP ({}): feature más importante = {} | plots guardados en {}",
                state.winner_family, top_feature, plot_dir)


def tune_optuna(state: SimpleNamespace, cfg: Config) -> dict:
    """Ronda larga de Optuna (post-feature-selection), solo sobre la familia ganadora."""
    return tune_optuna_family(
        state, cfg, state.winner_family,
        n_trials=cfg.optuna_n_trials, timeout=cfg.optuna_timeout_s,
        study_name=f"study_{state.level_str}_{state.target}_final_{state.winner_family}",
    )


def fit_final_model(state: SimpleNamespace, cfg: Config, evaluate_model, best_params: dict) -> None:
    family = MODEL_FAMILIES[state.winner_family]

    if best_params:
        label = f"{state.winner_family} (final, Optuna)"
    else:
        # run_optuna=False: en vez de un dict de hiperparámetros hardcodeado (como
        # antes), se reusan los hiperparámetros default del bench -- siempre están
        # disponibles porque el bench corre siempre (ver Config.run_bench_ml).
        best_params = state.bench_params[state.winner_family]
        label = f"{state.winner_family} (final, bench default)"

    # CV rolling-origin sobre train+valid para un n_estimators robusto (ver
    # rolling_cv_folds): el eval_metric acá es genérico (rmse/tweedie, igual que
    # tune_optuna), no state.wrmsse_metric -- ese está atado al valid_df original y
    # da resultados incorrectos sobre los folds más viejos. El último fold coincide
    # con el split actual (train=X_train, valid=X_valid); sus predicciones se
    # reusan para loggear la métrica "valid" (WRMSSE real) con evaluate_model.
    folds = rolling_cv_folds(state.X_train, state.y_train, state.X_valid, state.y_valid,
                              state.train["date"], state.valid_start, state.grain, cfg.cv_folds)

    def with_final_n_estimators(params):
        if not family.n_estimators_param:
            return params
        return {**params, family.n_estimators_param: cfg.final_n_estimators}

    t0 = time.perf_counter()
    best_iters, cv_fitted = [], None
    for X_tr, y_tr, X_val, y_val in folds:
        cv_fitted = family.fit(
            X_tr, y_tr, X_val, y_val, state.categorical_features, state.numerical_features,
            with_final_n_estimators(best_params), early_stopping_rounds=50, random_state=cfg.random_state,
        )
        best_iters.append(cv_fitted.best_iteration)
    valid_iters = [bi for bi in best_iters if bi is not None]
    n_estimators = round(np.mean(valid_iters)) if valid_iters else cfg.final_n_estimators
    logger.info("CV ({} folds) best_iteration: {} -> n_estimators final = {}",
                cfg.cv_folds, best_iters, n_estimators)
    evaluate_model(label, cv_fitted.predict(state.X_valid),
                    fit_time=time.perf_counter() - t0, category="ML", stage="final")

    model_params = best_params if not family.n_estimators_param else {
        **best_params, family.n_estimators_param: n_estimators,
    }
    X_trainval = pd.concat([state.X_train, state.X_valid])
    y_trainval = pd.concat([state.y_train, state.y_valid])
    final_fitted = family.fit(
        X_trainval, y_trainval, state.X_valid, state.y_valid,
        state.categorical_features, state.numerical_features,
        model_params, early_stopping_rounds=None, random_state=cfg.random_state,
    )
    fit_time = time.perf_counter() - t0

    metrics_test, df_pred = build_predictions_report(
        state.train, state.test, state.y_test, final_fitted.predict(state.X_test),
        target_col=state.target, m=state.m,
    )
    logger.info("Test WAPE: {:.2%} | Test WRMSSE: {:.4f}", metrics_test["wape"], metrics_test["wrmsse"])

    state.final_model = final_fitted
    state.model_params = model_params
    state.metrics_test = metrics_test
    state.df_pred = df_pred
