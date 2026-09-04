"""Entrena un modelo para un nivel/target, replicando el flujo de
notebooks/02_model.ipynb: split temporal, comparación de modelos base (naive,
estadísticos clásicos), bench de modelos ML (LightGBM/XGBoost/CatBoost/HistGB/Ridge,
cada uno tuneado con una ronda corta de Optuna), selección del modelo "ganador"
(mejor familia tree-based del bench), selección de features (permutation importance +
backward elimination) y explicación SHAP sobre el ganador, tuning más fino con Optuna
(ronda larga, post-feature-selection) y ajuste del modelo final.

El pipeline está separado en fases activables/desactivables (`Config.run_*`),
para poder saltear las que no hacen falta en una corrida dada (p.ej. iterar
sobre selección de features sin repetir el bench de modelos, o reentrenar el
modelo final con los hiperparámetros del bench sin correr la ronda larga de
Optuna de nuevo). `run_bench_ml` siempre tunea con Optuna (no hay modo
"hiperparámetros default"); `run_optuna` controla solo la ronda larga final.

Editar `CFG` (fases, hiperparámetros de tuning) y correr:
    make train-dataset ARGS="--levels 1,9,12"                 # todos los targets (sales + cada cumN)
    make train-dataset ARGS="--levels 12 --target cum28"      # un solo target, ver --target
    # o
    uv run python -m scripts.train_dataset --levels 1,9,12 --target cum28

Niveles con `split_by` (10-12: un dataset por combinación dept/store, ver
config/levels.py) entrenan un modelo por combinación, iterando sobre los
parquets ya generados por build-datasets (ver main()).

Salidas (target "sales" va directo en la carpeta base; cualquier otro target -- cumN --
en una subcarpeta propia, ver _artifacts_subdir(), para no mezclarse con los ~150
artifacts de "sales" ya generados):
    artifacts/results[/{target}]/{level}_{target}_model_comparison.csv   comparación de modelos,
                                                                            con columna `stage`
                                                                            (naive/statistical/bench/
                                                                            post_feature_selection/final)
    artifacts/results[/{target}]/{level}_{target}_series_metrics.csv     métricas por serie de esos modelos
    artifacts/results[/{target}]/{level}_{target}_feature_selection.csv  historial de backward elimination
    artifacts/plots[/{target}]/{level}_{target}_shap_*.png                explicación SHAP del ganador
    artifacts/optuna_study.db                                             estudios Optuna (sqlite, uno por
                                                                            level/grain/target/familia/etapa)
    artifacts/models[/{target}]/{level}_{target}_artifact.pkl            artifact liviano (modelo + features +
                                                                            métricas + winner_family, sin datos --
                                                                            ver src.data.split.reconstruct_test_data())
"""
import argparse
import gc
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from loguru import logger

import config
from src.data.split import load_data, rolling_cv_folds, split_data
from src.logging_setup import configure_logging
from src.evaluation import (
    build_all_series_metrics, build_predictions_report, evaluate_predictions, objective_metric,
)
from src.evaluation.scaled import clip_closed_stores
from src.modeling import (
    ALL_FAMILIES, MODEL_FAMILIES, TREE_FAMILIES,
    backward_feature_selection, compute_permutation_importance,
    drift, fit_ets, fit_prophet, fit_sarima, fit_tbats, fit_theta, historical_mean,
    moving_average, seasonal_naive,
)


# ============================== CONFIG ==============================
# Todo lo que controla la corrida vive acá, agrupado por fase. Editar y correr.

@dataclass
class Config:
    # --- selección de dataset ---
    level_id: int = 1
    target: str = "sales"  # "sales" o "cumN" (ver config.CUM_HORIZONS)

    # --- fases on/off ---
    run_baseline_naive: bool = True
    run_baseline_stats: bool = True    # SARIMA/ETS/Theta/TBATS/Prophet: serie x serie, lento
    # LightGBM/XGBoost/CatBoost/HistGB/Ridge, CADA UNO tuneado con una ronda corta de
    # Optuna (no es opcional -- si esta fase corre, todas las familias se tunean). El
    # ganador (mejor familia tree-based) sigue a feature selection / SHAP / Optuna final.
    run_bench_ml: bool = True
    run_feature_selection: bool = True  # permutation importance + backward elimination
    run_shap: bool = True
    run_optuna: bool = True             # ronda larga de Optuna, post-feature-selection, solo sobre el ganador
    save_artifact: bool = True

    # --- comparación de modelos base ---
    random_state: int = 42

    # --- selección de features ---
    permutation_sample_size: int = 15_000
    permutation_n_repeats: int = 5
    # Relativo al score actual (ver backward_feature_selection), no absoluto -- así
    # es comparable entre niveles con escalas de objective_metric muy distintas.
    # OJO: >0 hace que el criterio de aceptación "arrastre" (best_score se
    # actualiza a cada aceptación, incluso si es levemente peor que el anterior,
    # así que el degrade tolerado se acumula paso a paso sin techo) -- esto ya
    # pasaba con el algoritmo original uno-a-uno, no es nuevo del batching, pero
    # con tolerance>0 el ORDEN de las pruebas importa mucho más: batch vs.
    # uno-a-uno pueden terminar seleccionando conjuntos de features bien
    # distintos (probado: 6 vs. 21 features en un caso sintético). Por eso
    # se mantiene en 0.0 -- estricto, pero determinista y comparable entre
    # ambos modos. Subirlo es una decisión de selección de features, no de
    # performance, y hay que revisar el impacto real en cada nivel.
    backward_tolerance: float = 0.0
    # Cuántas features candidatas se prueba remover juntas en cada reentrenamiento
    # de backward_feature_selection (ver docstring ahí, sección "Batching"). 1 =
    # una por una (algoritmo original, más lento con muchas features); >1 = por
    # lotes, con fallback automático a uno-a-uno si el lote se rechaza. Más rápido,
    # y en la práctica tiende a podar mejor grupos de features redundantes entre
    # sí -- pero la selección final puede diferir de la de max_batch_size=1 (no
    # es un cambio de criterio de aceptación, es que un bloque completo es una
    # prueba más exigente que sus features por separado). Para reproducir bit a
    # bit una corrida sin batching, usar 1.
    feature_selection_batch_size: int = 8

    # --- SHAP ---
    shap_sample_size: int = 300_000

    # --- Optuna: bench (TODAS las familias de ALL_FAMILIES, corto) ---
    optuna_bench_n_trials: int = 200
    optuna_bench_timeout_s: int = 5 * 60

    # --- Optuna: final (largo, solo sobre la familia ganadora, post-feature-selection) ---
    optuna_n_trials: int = 500
    optuna_timeout_s: int = 10 * 60
    optuna_n_estimators: int = 1_500

    # --- modelo final ---
    # Folds rolling-origin sobre train+valid para elegir n_estimators antes del
    # refit final sobre todos los datos (ver rolling_cv_folds). 1 = sin CV, un solo
    # split como antes.
    cv_folds: int = 3
    final_n_estimators: int = 5_000
    objective_by_level: dict = field(default_factory=lambda: {12: "tweedie"})
    default_objective: str = "regression_l2"

    @property
    def objective(self) -> str:
        return self.objective_by_level.get(self.level_id, self.default_objective)

    # Fijo si run_optuna=False (fit_final_model cae al fallback de bench_params en
    # ese caso, ver abajo); si run_optuna=True y objective=="tweedie", Optuna
    # tunea este valor (1.1-1.9) para las familias que lo soportan (lightgbm/xgboost)
    # en vez de usar el fijo.
    tweedie_variance_power: float = 1.5


CFG = Config()


def _artifacts_subdir(base: Path, target: str) -> Path:
    """Subcarpeta por target salvo 'sales': deja los artifacts de cum7/14/21/28 (y
    cualquier otro target futuro) separados de los ~150 archivos de 'sales' ya
    generados en `base` sin tocarlos ni requerir migrarlos."""
    return base if target == "sales" else base / target


def _update_registry(key: str, version: str, path: Path, artifact: dict) -> None:
    """Anota una versión nueva en artifacts/models/registry.json (read-modify-write,
    sin locking: entrenamiento local de a uno por vez, igual que hoy). No reemplaza
    al archivo plano `{level_str}_{target}_artifact.pkl` (ese lo sigue leyendo
    notebooks/03_predictions.ipynb tal cual) -- el registry es solo para que la API
    (api/) pueda listar/servir versiones puntuales."""
    registry_path = config.MODELS_DIR / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    entry = registry.setdefault(key, {"latest": None, "versions": {}})
    entry["versions"][version] = {
        "path": str(path.relative_to(config.ROOT)),
        "trained_at": version,
        "winner_family": artifact.get("winner_family"),
        "wape_test": artifact["wape_test"],
        "wrmsse_test": artifact["wrmsse_test"],
    }
    entry["latest"] = version
    registry_path.write_text(json.dumps(registry, indent=2))


# ============================== FASES ==============================

def make_evaluator(state: SimpleNamespace, cfg: Config):
    """Cierra sobre `state`/`cfg` para acumular cada modelo evaluado en
    `model_results`/`predictions_valid`, igual que `evaluate_model` en el notebook.
    Calcula `objective_score` (rmse o deviance de Tweedie, según `cfg.objective`) para
    cada modelo -- es la métrica decisional real (ver `objective_metric`), la que usa
    `pick_winner` para elegir el ganador entre familias (comparar por `rmse`/`wrmsse`
    directamente sería incorrecto en niveles tweedie)."""
    def evaluate_model(name, y_pred_valid, fit_time=None, category=None, stage=None):
        y_pred_clipped = clip_closed_stores(state.valid, y_pred_valid)
        objective_score = objective_metric(state.y_valid, y_pred_clipped, cfg.objective, cfg.tweedie_variance_power)
        result = evaluate_predictions(
            state.train, state.valid, state.y_valid, y_pred_valid, name,
            fit_time=fit_time, category=category, stage=stage, objective_score=objective_score,
            target_col=state.target, m=state.m,
        )
        state.model_results.append(result)
        state.predictions_valid[name] = y_pred_valid
        return result

    return evaluate_model


def run_baseline_naive(state: SimpleNamespace, cfg: Config, evaluate_model) -> None:
    grain_letter = state.grain[0]

    for window in config.SN_WINDOWS[state.grain]:
        t0 = time.perf_counter()
        y_pred = seasonal_naive(state.train, state.valid, state.target, season_length=window)
        evaluate_model(f"Seasonal naive ({window}{grain_letter})", y_pred,
                        fit_time=time.perf_counter() - t0, category="Naive", stage="naive")

    t0 = time.perf_counter()
    evaluate_model("Drift", drift(state.train, state.valid, state.target),
                    fit_time=time.perf_counter() - t0, category="Naive", stage="naive")

    t0 = time.perf_counter()
    evaluate_model("Historical mean", historical_mean(state.train, state.valid, state.target),
                    fit_time=time.perf_counter() - t0, category="Naive", stage="naive")

    for window in config.MA_WINDOWS[state.grain]:
        t0 = time.perf_counter()
        y_pred = moving_average(state.train, state.valid, state.target, window=window)
        evaluate_model(f"Moving average ({window}{grain_letter})", y_pred,
                        fit_time=time.perf_counter() - t0, category="Naive", stage="naive")


def run_baseline_stats(state: SimpleNamespace, evaluate_model) -> None:
    season_length = config.SEASON_LENGTH[state.grain]

    for name, fn, kwargs in [
        ("SARIMA", fit_sarima, dict(seasonal_order=(1, 1, 1, season_length))),
        ("ETS (Holt-Winters)", fit_ets, dict(seasonal_periods=season_length)),
        ("Theta", fit_theta, dict(period=season_length)),
        ("TBATS", fit_tbats, dict(season_length=(season_length,))),
        ("Prophet", fit_prophet, dict(weekly_seasonality=(state.grain == "daily"))),
    ]:
        t0 = time.perf_counter()
        y_pred = fn(state.train, state.valid, state.target, **kwargs)
        evaluate_model(name, y_pred, fit_time=time.perf_counter() - t0, category="Statistical", stage="statistical")


def tune_optuna_family(state: SimpleNamespace, cfg: Config, family_name: str,
                        n_trials: int, timeout: int, study_name: str) -> dict:
    """Ronda de Optuna para una familia de modelo puntual: tunea `family.optuna_param_space`
    contra `objective_metric` (rmse/tweedie deviance, coherente con `cfg.objective` --
    el WRMSSE se guarda como user_attr solo para informar, igual que antes). Usada tanto
    por `run_bench_ml` (corta, todas las familias) como por `tune_optuna` (larga, solo
    el ganador, post-feature-selection).

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
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # user_attrs["wrmsse"] puede faltar en trials de corridas viejas del mismo
    # study persistido en sqlite (load_if_exists=True mezcla historial entre
    # ejecuciones) -- es solo informativo, no participa en qué hiperparámetros
    # elige Optuna (eso sale de study.best_trial.params).
    best_wrmsse = study.best_trial.user_attrs.get("wrmsse")
    wrmsse_str = f"{best_wrmsse:.4f}" if best_wrmsse is not None else "N/A (trial de una corrida anterior sin este dato)"
    logger.info("[{}] Mejor {} (Optuna): {:.4f} | WRMSSE: {} (informativo)",
                family_name, cfg.objective, study.best_trial.value, wrmsse_str)
    logger.info("[{}] Mejores hiperparámetros: {}", family_name, study.best_trial.params)

    return {**fixed_params, **study.best_trial.params}


def run_bench_ml(state: SimpleNamespace, cfg: Config, evaluate_model) -> None:
    """Reemplaza a la vieja comparación de hiperparámetros default: cada familia de
    `ALL_FAMILIES` se tunea con una ronda corta de Optuna (`optuna_bench_*`) y se
    evalúa ya con sus mejores hiperparámetros encontrados. `pick_winner` decide el
    ganador entre estos resultados (solo familias tree-based, ver TREE_FAMILIES)."""
    state.bench_models = {}
    state.bench_params = {}
    state.bench_scores = {}

    for family_name in ALL_FAMILIES:
        t0 = time.perf_counter()
        best_params = tune_optuna_family(
            state, cfg, family_name,
            n_trials=cfg.optuna_bench_n_trials, timeout=cfg.optuna_bench_timeout_s,
            study_name=f"study_{state.level_str}_{state.target}_bench_{family_name}",
        )
        fitted = MODEL_FAMILIES[family_name].fit(
            state.X_train, state.y_train, state.X_valid, state.y_valid,
            state.categorical_features, state.numerical_features,
            best_params, early_stopping_rounds=30, random_state=cfg.random_state,
        )
        fit_time = time.perf_counter() - t0
        result = evaluate_model(f"{family_name} (Optuna bench)", fitted.predict(state.X_valid),
                                 fit_time=fit_time, category="ML", stage="bench")

        state.bench_models[family_name] = fitted
        state.bench_params[family_name] = best_params
        state.bench_scores[family_name] = result["objective_score"]


def pick_winner(state: SimpleNamespace, cfg: Config) -> str:
    """Elige la mejor familia tree-based del bench (por `objective_score`, no
    `rmse`/`wrmsse` directamente -- coherente en niveles tweedie). Ridge participa del
    bench pero nunca puede ganar (no es compatible con `shap.TreeExplainer`)."""
    winner = min(TREE_FAMILIES, key=lambda f: state.bench_scores[f])
    state.winner_family = winner
    logger.success("Modelo ganador (bench, {} familias tree-based comparadas): {} | {}={:.4f}",
                    len(TREE_FAMILIES), winner, cfg.objective, state.bench_scores[winner])
    return winner


def save_model_comparison(state: SimpleNamespace) -> None:
    if not state.model_results:
        return
    out_dir = _artifacts_subdir(config.ARTIFACTS_DIR / "results", state.target)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(state.model_results).sort_values("wrmsse")
    results_df.to_csv(out_dir / f"{state.level_str}_{state.target}_model_comparison.csv", index=False)

    if state.predictions_valid:
        series_metrics_df = build_all_series_metrics(
            state.train, state.valid, state.y_valid, state.predictions_valid,
            target_col=state.target, m=state.m,
        )
        series_metrics_df.to_csv(out_dir / f"{state.level_str}_{state.target}_series_metrics.csv", index=False)

    logger.info("Comparación de modelos guardada en {}", out_dir)


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

    out_dir = _artifacts_subdir(config.ARTIFACTS_DIR / "results", state.target)
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
    evaluate_model(f"{state.winner_family} (Optuna bench, selected features)", fitted.predict(state.X_valid),
                    fit_time=fit_time, category="ML", stage="post_feature_selection")

    explainer = shap.TreeExplainer(fitted.estimator)
    X_shap_raw = state.X_test.sample(n=min(cfg.shap_sample_size, len(state.X_test)), random_state=cfg.random_state)
    X_shap = fitted.prepare(X_shap_raw)
    shap_values = explainer.shap_values(X_shap)

    plot_dir = _artifacts_subdir(config.ARTIFACTS_DIR / "plots", state.target)
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
        # antes), se reusan los ya tuneados en el bench (ronda corta) -- siempre están
        # disponibles porque el bench corre siempre (ver Config.run_bench_ml).
        best_params = state.bench_params[state.winner_family]
        label = f"{state.winner_family} (final, bench Optuna)"

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


def export_artifact(state: SimpleNamespace, cfg: Config) -> None:
    results_df = pd.DataFrame(state.model_results).sort_values("wrmsse")

    family = MODEL_FAMILIES[state.winner_family]
    feature_importance = family.gain_importance(state.final_model, state.features)
    if feature_importance is None:
        # Familias sin importancia nativa (p.ej. histgb): permutation importance sobre
        # validación, misma función que usa select_features (ya model-agnostic).
        feature_importance = compute_permutation_importance(
            state.final_model, state.X_valid, state.y_valid, state.features,
            sample_size=cfg.permutation_sample_size, n_repeats=cfg.permutation_n_repeats,
            random_state=cfg.random_state, objective=cfg.objective, tweedie_variance_power=cfg.tweedie_variance_power,
        )

    # Solo modelo + metadata (features, métricas): liviano para respaldar/mover entre
    # máquinas sin arrastrar datos. df/X_*/y_*/train/valid/test NO se guardan --
    # src.data.split.reconstruct_test_data() las recrea desde el parquet en
    # data/datasets/ repitiendo load_data()+split_data() (ver notebooks/03_predictions.ipynb).
    artifact = {
        "level": state.level_str,
        "level_id": state.level.id,
        "winner_family": state.winner_family,
        "model": state.final_model,
        "model_params": state.model_params,
        "features": state.features,
        "categorical_features": state.categorical_features,
        "numerical_features": state.numerical_features,
        "id_cols": state.id_cols,
        "leaky_cols": state.leaky_cols,
        "target": state.target,
        "feature_importance": feature_importance,
        "model_results": state.model_results,
        "results_df": results_df,
        "wape_valid": state.model_results[-1]["wape"],
        "wrmsse_valid": state.model_results[-1]["wrmsse"],
        "wape_test": state.metrics_test["wape"],
        "wrmsse_test": state.metrics_test["wrmsse"],
        "train_start": str(state.first_date.date()),
        "valid_start": str(state.valid_start.date()),
        "test_start": str(state.test_start.date()),
    }

    out_dir = _artifacts_subdir(config.MODELS_DIR, state.target)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{state.level_str}_{state.target}_artifact.pkl"
    joblib.dump(artifact, artifact_path)
    logger.success("Artifact guardado en {} (modelo: {})", artifact_path, state.winner_family)

    # Copia versionada + registry.json: permite a api/ servir una versión puntual y
    # conservar el historial, sin tocar el flujo del archivo plano de arriba.
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versions_dir = out_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = versions_dir / f"{state.level_str}_{state.target}_artifact_{version}.pkl"
    joblib.dump(artifact, versioned_path)
    _update_registry(f"{state.level_str}_{state.target}", version, versioned_path, artifact)
    logger.success("Versión {} registrada en registry.json", version)


def run_pipeline(cfg: Config, dataset_path: Path | None = None) -> None:
    t_start = time.perf_counter()

    state = load_data(cfg.level_id, cfg.target, dataset_path)
    split_data(state, cfg.target)
    evaluate_model = make_evaluator(state, cfg)

    if cfg.run_baseline_naive:
        t0 = time.perf_counter()
        run_baseline_naive(state, cfg, evaluate_model)
        logger.info("Fase naive: {:.1f}s", time.perf_counter() - t0)

    if cfg.run_baseline_stats:
        t0 = time.perf_counter()
        run_baseline_stats(state, evaluate_model)
        logger.info("Fase estadísticos clásicos: {:.1f}s", time.perf_counter() - t0)

    if cfg.run_bench_ml:
        t0 = time.perf_counter()
        run_bench_ml(state, cfg, evaluate_model)
        pick_winner(state, cfg)
        logger.info("Fase bench ML (Optuna corto, {} familias): {:.1f}s",
                    len(ALL_FAMILIES), time.perf_counter() - t0)

    if not hasattr(state, "winner_family"):
        logger.warning("run_bench_ml=False: no hay modelo ganador, se saltea el resto del pipeline ML.")
        save_model_comparison(state)
        return

    if cfg.run_feature_selection:
        t0 = time.perf_counter()
        select_features(state, cfg)
        logger.info("Fase selección de features: {:.1f}s", time.perf_counter() - t0)

    if cfg.run_shap:
        t0 = time.perf_counter()
        run_shap(state, cfg, evaluate_model)
        logger.info("Fase SHAP: {:.1f}s", time.perf_counter() - t0)

    best_params = {}
    if cfg.run_optuna:
        t0 = time.perf_counter()
        best_params = tune_optuna(state, cfg)
        logger.info("Fase Optuna final: {:.1f}s", time.perf_counter() - t0)

    fit_final_model(state, cfg, evaluate_model, best_params)
    save_model_comparison(state)

    if cfg.save_artifact:
        export_artifact(state, cfg)

    logger.info("Pipeline completo ({}/{}) en {:.1f}s", state.level_str, state.target,
                time.perf_counter() - t_start)


def main():
    ap = argparse.ArgumentParser(description="Entrena modelos ML (bench Optuna + ganador) por nivel de agregación M5.")
    ap.add_argument("--levels", help="IDs separados por coma, p.ej. 1,9,12. "
                    "Default: config.ACTIVE_LEVEL_IDS.")
    ap.add_argument("--target", default=None,
                    help='Target: "sales" o "cumN" (ver config.CUM_EVAL_HORIZONS, p.ej. "cum28"). '
                         'Default: todos -- "sales" + cada cumN en config.CUM_EVAL_HORIZONS.')
    args = ap.parse_args()

    level_ids = [int(x) for x in args.levels.split(",")] if args.levels else list(config.ACTIVE_LEVEL_IDS)
    targets = [args.target] if args.target else ["sales", *(f"cum{n}" for n in config.CUM_EVAL_HORIZONS)]

    for target in targets:
        for level_id in level_ids:
            level = config.LEVELS_BY_ID[level_id]
            cfg = replace(CFG, level_id=level_id, target=target)

            if not level.split_by:
                try:
                    run_pipeline(cfg)
                except Exception:
                    logger.exception("Nivel {} target {} falló, sigue con el resto.", level_id, target)
                gc.collect()
                continue

            # split_by (10-12): un modelo por combinación (p.ej. dept_id x store_id en L12),
            # un parquet por combinación ya generado por build-datasets.
            combos = config.featured_level_combos(level, level.grains[0])
            if not combos:
                logger.warning("Nivel {} ({}): no hay parquets de combinación en data/datasets/{}/ "
                                "-- correr build-datasets primero. Se saltea.",
                                level_id, level.name, level.grains[0])
                continue
            logger.info("Nivel {} ({}): {} combinaciones", level_id, level.name, len(combos))
            for path in combos:
                try:
                    run_pipeline(cfg, dataset_path=path)
                except Exception:
                    logger.exception("Nivel {} combo {} target {} falló, sigue con el resto.", level_id, path.stem, target)
                gc.collect()


if __name__ == "__main__":
    configure_logging("train_dataset")
    main()
