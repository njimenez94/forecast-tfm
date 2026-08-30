"""Entrena un modelo LightGBM para un nivel/target, replicando el flujo de
notebooks/02_model.ipynb: split temporal, comparación de modelos base (naive,
estadísticos clásicos, ML con hiperparámetros default), selección de features
(permutation importance + backward elimination), explicación SHAP, tuning con
Optuna y ajuste del modelo final.

El pipeline está separado en fases activables/desactivables (`Config.run_*`),
para poder saltear las que no hacen falta en una corrida dada (p.ej. iterar
sobre selección de features sin repetir la comparación de modelos base, o
reentrenar el modelo final con `CFG.default_lgbm_params` sin correr Optuna de nuevo).

Editar `CFG` (fases, hiperparámetros de tuning) y correr:
    make train-dataset ARGS="--levels 1,9,12"                 # target "sales" (CFG.target)
    make train-dataset ARGS="--levels 12 --target cum28"      # target cumN, ver --target
    # o
    uv run python -m scripts.train_dataset --levels 1,9,12 --target cum28

Niveles con `split_by` (10-12: un dataset por combinación dept/store, ver
config/levels.py) entrenan un modelo por combinación, iterando sobre los
parquets ya generados por build-datasets (ver main()).

Salidas (target "sales" va directo en la carpeta base; cualquier otro target -- cumN --
en una subcarpeta propia, ver _artifacts_subdir(), para no mezclarse con los ~150
artifacts de "sales" ya generados):
    artifacts/results[/{target}]/{level}_{target}_model_comparison.csv   comparación de modelos (si corrió alguna fase de baseline)
    artifacts/results[/{target}]/{level}_{target}_series_metrics.csv     métricas por serie de esos modelos
    artifacts/results[/{target}]/{level}_{target}_feature_selection.csv  historial de backward elimination
    artifacts/plots[/{target}]/{level}_{target}_shap_*.png                explicación SHAP del modelo simple
    artifacts/optuna_study.db                                             estudio Optuna (sqlite, uno por level/grain/target)
    artifacts/models[/{target}]/{level}_{target}_artifact.pkl            artifact liviano (modelo + features + métricas,
                                                                            sin datos -- ver src.data.split.reconstruct_test_data())
"""
import argparse
import gc
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from loguru import logger

import config
from src.data.split import load_data, rolling_cv_folds, split_data
from src.evaluation import (
    build_all_series_metrics, build_predictions_report, evaluate_predictions, objective_metric,
)
from src.modeling import (
    backward_feature_selection, catboost_features, compute_permutation_importance,
    drift, fit_catboost, fit_ets, fit_histgb, fit_lightgbm, fit_prophet, fit_ridge,
    fit_sarima, fit_tbats, fit_theta, fit_xgboost, histgb_features, historical_mean,
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
    run_baseline_ml: bool = True       # LightGBM/XGBoost/CatBoost/HistGB/Ridge, hiperparámetros default
    run_feature_selection: bool = True  # permutation importance + backward elimination
    run_shap: bool = True
    run_optuna: bool = True
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

    # --- Optuna ---
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

    # Fijo si run_optuna=False; si run_optuna=True y objective=="tweedie", Optuna
    # tunea este valor (1.1-1.9) en vez de usar el fijo.
    tweedie_variance_power: float = 1.5
    # Hiperparámetros del modelo final cuando NO corre Optuna (fallback, sin tunear).
    default_lgbm_params: dict = field(default_factory=lambda: {
        "metric": "mae",
        "learning_rate": 0.05,
        "n_estimators": 200,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "n_jobs": -1,
    })


CFG = Config()


def _artifacts_subdir(base: Path, target: str) -> Path:
    """Subcarpeta por target salvo 'sales': deja los artifacts de cum7/14/21/28 (y
    cualquier otro target futuro) separados de los ~150 archivos de 'sales' ya
    generados en `base` sin tocarlos ni requerir migrarlos."""
    return base if target == "sales" else base / target


# ============================== FASES ==============================

def make_evaluator(state: SimpleNamespace):
    """Cierra sobre `state` para acumular cada modelo evaluado en
    `model_results`/`predictions_valid`, igual que `evaluate_model` en el notebook."""
    def evaluate_model(name, y_pred_valid, fit_time=None, category=None):
        result = evaluate_predictions(
            state.train, state.valid, state.y_valid, y_pred_valid, name,
            fit_time=fit_time, category=category, target_col=state.target, m=state.m,
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
                        fit_time=time.perf_counter() - t0, category="Naive")

    t0 = time.perf_counter()
    evaluate_model("Drift", drift(state.train, state.valid, state.target),
                    fit_time=time.perf_counter() - t0, category="Naive")

    t0 = time.perf_counter()
    evaluate_model("Historical mean", historical_mean(state.train, state.valid, state.target),
                    fit_time=time.perf_counter() - t0, category="Naive")

    for window in config.MA_WINDOWS[state.grain]:
        t0 = time.perf_counter()
        y_pred = moving_average(state.train, state.valid, state.target, window=window)
        evaluate_model(f"Moving average ({window}{grain_letter})", y_pred,
                        fit_time=time.perf_counter() - t0, category="Naive")


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
        evaluate_model(name, y_pred, fit_time=time.perf_counter() - t0, category="Statistical")


def run_baseline_ml(state: SimpleNamespace, cfg: Config, evaluate_model) -> None:
    t0 = time.perf_counter()
    lgbm_model = fit_lightgbm(state.X_train, state.y_train, state.categorical_features,
                               random_state=cfg.random_state, **resolve_objective(cfg))
    evaluate_model("LightGBM", lgbm_model.predict(state.X_valid),
                    fit_time=time.perf_counter() - t0, category="ML")
    state.lgbm_model = lgbm_model

    t0 = time.perf_counter()
    xgb_model = fit_xgboost(state.X_train, state.y_train, random_state=cfg.random_state)
    evaluate_model("XGBoost", xgb_model.predict(state.X_valid),
                    fit_time=time.perf_counter() - t0, category="ML")

    t0 = time.perf_counter()
    catboost_model = fit_catboost(state.X_train, state.y_train, state.categorical_features,
                                   random_state=cfg.random_state)
    X_valid_cb = catboost_features(state.X_valid, state.categorical_features)
    evaluate_model("CatBoost", catboost_model.predict(X_valid_cb),
                    fit_time=time.perf_counter() - t0, category="ML")

    t0 = time.perf_counter()
    hgb_model = fit_histgb(state.X_train, state.y_train, random_state=cfg.random_state)
    X_valid_hgb = histgb_features(state.X_valid, hgb_model.high_cardinality_features_)
    evaluate_model("HistGradientBoosting", hgb_model.predict(X_valid_hgb),
                    fit_time=time.perf_counter() - t0, category="ML")

    t0 = time.perf_counter()
    ridge_model = fit_ridge(state.X_train, state.y_train, state.numerical_features,
                             random_state=cfg.random_state)
    evaluate_model("Ridge", ridge_model.predict(state.X_valid[state.numerical_features]),
                    fit_time=time.perf_counter() - t0, category="ML")


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
    objective_kwargs = resolve_objective(cfg)

    lgbm_model = getattr(state, "lgbm_model", None)
    if lgbm_model is None:
        lgbm_model = fit_lightgbm(state.X_train, state.y_train, state.categorical_features,
                                   random_state=cfg.random_state, **objective_kwargs)

    importance_gain = pd.Series(
        lgbm_model.booster_.feature_importance(importance_type="gain"), index=state.features,
    ).sort_values(ascending=False)
    logger.info("Top 10 gain importance:\n{}", importance_gain.head(10))

    importance_perm = compute_permutation_importance(
        lgbm_model, state.X_valid, state.y_valid, state.features,
        sample_size=cfg.permutation_sample_size, n_repeats=cfg.permutation_n_repeats,
        random_state=cfg.random_state, **objective_kwargs,
    )
    logger.info("Top 10 permutation importance:\n{}", importance_perm.head(10))

    selected_features, final_score, final_wrmsse, selection_log = backward_feature_selection(
        state.X_train, state.y_train, state.X_valid, state.y_valid, state.train, state.valid,
        state.features, state.categorical_features, importance_perm,
        tolerance=cfg.backward_tolerance, random_state=cfg.random_state,
        max_batch_size=cfg.feature_selection_batch_size, **objective_kwargs,
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
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import shap
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**resolve_objective(cfg))
    t0 = time.perf_counter()
    model.fit(
        state.X_train, state.y_train,
        eval_set=[(state.X_valid, state.y_valid)],
        eval_metric=[state.wrmsse_metric],
        callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(10)],
    )
    fit_time = time.perf_counter() - t0
    evaluate_model("LightGBM (selected features)", model.predict(state.X_valid),
                    fit_time=fit_time, category="ML")

    explainer = shap.TreeExplainer(model)
    X_shap = state.X_test.sample(n=min(cfg.shap_sample_size, len(state.X_test)), random_state=cfg.random_state)
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

    logger.info("SHAP: feature más importante = {} | plots guardados en {}", top_feature, plot_dir)


def resolve_objective(cfg: Config) -> dict:
    """Objective de `cfg.objective`, mismo criterio con o sin Optuna."""
    kwargs = {"objective": cfg.objective}
    if cfg.objective == "tweedie":
        kwargs["tweedie_variance_power"] = cfg.tweedie_variance_power
    return kwargs


def tune_optuna(state: SimpleNamespace, cfg: Config) -> dict:
    import lightgbm as lgb
    import optuna
    from optuna.integration import LightGBMPruningCallback

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective_kwargs = resolve_objective(cfg)
    eval_metric = "tweedie" if cfg.objective == "tweedie" else "rmse"

    def objective(trial):
        params = dict(
            **objective_kwargs,
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            n_estimators=cfg.optuna_n_estimators,
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            max_depth=trial.suggest_int("max_depth", 5, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5, log=True),
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        if cfg.objective == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            state.X_train, state.y_train,
            eval_set=[(state.X_valid, state.y_valid)],
            eval_metric=eval_metric,
            categorical_feature=state.categorical_features,
            callbacks=[
                lgb.early_stopping(30, first_metric_only=True, verbose=False),
                LightGBMPruningCallback(trial, eval_metric),
            ],
        )
        y_pred = model.predict(state.X_valid, num_iteration=model.best_iteration_)
        # Optuna decide por `objective_metric` (coherente con cfg.objective); el
        # WRMSSE se calcula y se guarda como user_attr solo para informar.
        tweedie_power = params.get("tweedie_variance_power", cfg.tweedie_variance_power)
        score = objective_metric(state.y_valid, y_pred, cfg.objective, tweedie_power)
        _, final_wrmsse, _ = state.wrmsse_metric(state.y_valid, y_pred)
        trial.set_user_attr("wrmsse", final_wrmsse)
        return score

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"study_{state.level_str}_{state.target}",
        direction="minimize",
        storage=f"sqlite:///{config.ARTIFACTS_DIR / 'optuna_study.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=cfg.optuna_n_trials, timeout=cfg.optuna_timeout_s,
                    show_progress_bar=True)

    logger.info("Mejor {} (Optuna): {:.4f} | WRMSSE: {:.4f} (informativo)",
                cfg.objective, study.best_trial.value, study.best_trial.user_attrs["wrmsse"])
    logger.info("Mejores hiperparámetros: {}", study.best_trial.params)
    return study.best_trial.params


def fit_final_model(state: SimpleNamespace, cfg: Config, evaluate_model, best_params: dict) -> None:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    objective_kwargs = resolve_objective(cfg)

    if best_params:
        # best_params puede traer tweedie_variance_power tuneado (ver tune_optuna) --
        # va después de objective_kwargs para que gane sobre el valor fijo de cfg.
        model_params = {
            **objective_kwargs, **best_params,
            "random_state": cfg.random_state, "n_jobs": -1, "verbosity": -1, "subsample_freq": 1,
        }
        label = "LightGBM (final, Optuna)"
    else:
        model_params = {**cfg.default_lgbm_params, "seed": cfg.random_state, **objective_kwargs}
        label = "LightGBM (final, params default)"

    # CV rolling-origin sobre train+valid para un n_estimators robusto (ver
    # rolling_cv_folds): el eval_metric acá es genérico (rmse/tweedie, igual que
    # tune_optuna), no state.wrmsse_metric -- ese está atado al valid_df original y
    # da resultados incorrectos sobre los folds más viejos. El último fold coincide
    # con el split actual (train=X_train, valid=X_valid); sus predicciones se
    # reusan para loggear la métrica "valid" (WRMSSE real) con evaluate_model.
    folds = rolling_cv_folds(state.X_train, state.y_train, state.X_valid, state.y_valid,
                              state.train["date"], state.valid_start, state.grain, cfg.cv_folds)
    eval_metric = "tweedie" if cfg.objective == "tweedie" else "rmse"

    t0 = time.perf_counter()
    best_iters, cv_model = [], None
    for X_tr, y_tr, X_val, y_val in folds:
        cv_model = LGBMRegressor(**{**model_params, "n_estimators": cfg.final_n_estimators})
        cv_model.fit(
            X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=eval_metric,
            categorical_feature=state.categorical_features,
            callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
        )
        best_iters.append(cv_model.best_iteration_)
    n_estimators = round(np.mean(best_iters))
    logger.info("CV ({} folds) best_iteration: {} -> n_estimators final = {}",
                cfg.cv_folds, best_iters, n_estimators)
    evaluate_model(label, cv_model.predict(state.X_valid, num_iteration=cv_model.best_iteration_),
                    fit_time=time.perf_counter() - t0, category="ML")

    model_params = {**model_params, "n_estimators": n_estimators}
    final_model = LGBMRegressor(**model_params)
    X_trainval = pd.concat([state.X_train, state.X_valid])
    y_trainval = pd.concat([state.y_train, state.y_valid])
    final_model.fit(X_trainval, y_trainval, categorical_feature=state.categorical_features)
    fit_time = time.perf_counter() - t0

    metrics_test, df_pred = build_predictions_report(
        state.train, state.test, state.y_test, final_model.predict(state.X_test),
        target_col=state.target, m=state.m,
    )
    logger.info("Test WAPE: {:.2%} | Test WRMSSE: {:.4f}", metrics_test["wape"], metrics_test["wrmsse"])

    state.final_model = final_model
    state.model_params = model_params
    state.metrics_test = metrics_test
    state.df_pred = df_pred


def export_artifact(state: SimpleNamespace) -> None:
    results_df = pd.DataFrame(state.model_results).sort_values("wrmsse")

    feature_importance = pd.Series(
        state.final_model.booster_.feature_importance(importance_type="gain"), index=state.features,
    ).sort_values(ascending=False)

    # Solo modelo + metadata (features, métricas): liviano para respaldar/mover entre
    # máquinas sin arrastrar datos. df/X_*/y_*/train/valid/test NO se guardan --
    # src.data.split.reconstruct_test_data() las recrea desde el parquet en
    # data/datasets/ repitiendo load_data()+split_data() (ver notebooks/03_predictions.ipynb).
    artifact = {
        "level": state.level_str,
        "level_id": state.level.id,
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
    logger.success("Artifact guardado en {}", artifact_path)


def run_pipeline(cfg: Config, dataset_path: Path | None = None) -> None:
    t_start = time.perf_counter()

    state = load_data(cfg.level_id, cfg.target, dataset_path)
    split_data(state, cfg.target)
    evaluate_model = make_evaluator(state)

    if cfg.run_baseline_naive:
        t0 = time.perf_counter()
        run_baseline_naive(state, cfg, evaluate_model)
        logger.info("Fase naive: {:.1f}s", time.perf_counter() - t0)

    if cfg.run_baseline_stats:
        t0 = time.perf_counter()
        run_baseline_stats(state, evaluate_model)
        logger.info("Fase estadísticos clásicos: {:.1f}s", time.perf_counter() - t0)

    if cfg.run_baseline_ml:
        t0 = time.perf_counter()
        run_baseline_ml(state, cfg, evaluate_model)
        logger.info("Fase ML (hiperparámetros default): {:.1f}s", time.perf_counter() - t0)

    save_model_comparison(state)

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
        logger.info("Fase Optuna: {:.1f}s", time.perf_counter() - t0)

    fit_final_model(state, cfg, evaluate_model, best_params)

    if cfg.save_artifact:
        export_artifact(state)

    logger.info("Pipeline completo ({}/{}) en {:.1f}s", state.level_str, state.target,
                time.perf_counter() - t_start)


def main():
    ap = argparse.ArgumentParser(description="Entrena LightGBM por nivel de agregación M5.")
    ap.add_argument("--levels", help="IDs separados por coma, p.ej. 1,9,12. "
                    "Default: config.ACTIVE_LEVEL_IDS.")
    ap.add_argument("--target", default=CFG.target,
                    help='Target: "sales" o "cumN" (ver config.CUM_EVAL_HORIZONS, p.ej. "cum28"). '
                         f"Default: {CFG.target!r}.")
    args = ap.parse_args()

    level_ids = [int(x) for x in args.levels.split(",")] if args.levels else list(config.ACTIVE_LEVEL_IDS)

    for level_id in level_ids:
        level = config.LEVELS_BY_ID[level_id]
        cfg = replace(CFG, level_id=level_id, target=args.target)

        if not level.split_by:
            try:
                run_pipeline(cfg)
            except Exception:
                logger.exception("Nivel {} falló, sigue con el resto.", level_id)
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
                logger.exception("Nivel {} combo {} falló, sigue con el resto.", level_id, path.stem)
            gc.collect()


if __name__ == "__main__":
    main()
