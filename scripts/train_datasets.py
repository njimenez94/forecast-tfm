"""Entrena un modelo LightGBM para un nivel/target, replicando el flujo de
notebooks/02_model.ipynb: split temporal, comparación de modelos base (naive,
estadísticos clásicos, ML con hiperparámetros default), selección de features
(permutation importance + backward elimination), explicación SHAP, tuning con
Optuna y ajuste del modelo final.

El pipeline está separado en fases activables/desactivables (`Config.run_*`),
para poder saltear las que no hacen falta en una corrida dada (p.ej. iterar
sobre selección de features sin repetir la comparación de modelos base, o
reentrenar el modelo final con `CFG.default_lgbm_params` sin correr Optuna de nuevo).

Editar `CFG` (nivel, target, fases, hiperparámetros de tuning) y correr:
    make train-datasets
    # o
    uv run python -m scripts.train_datasets

Salidas:
    artifacts/results/{level}_{target}_model_comparison.csv   comparación de modelos (si corrió alguna fase de baseline)
    artifacts/results/{level}_{target}_series_metrics.csv     métricas por serie de esos modelos
    artifacts/results/{level}_{target}_feature_selection.csv  historial de backward elimination
    artifacts/plots/{level}_{target}_shap_*.png                explicación SHAP del modelo simple
    artifacts/optuna_study.db                                  estudio Optuna (sqlite, uno por level/grain/target)
    artifacts/models/{level}_{target}_artifact.pkl             artifact final (modelo + datos + métricas)
"""
import gc
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from loguru import logger

import config
from src.data import build_feature_matrices, date_split
from src.evaluation import (
    build_all_series_metrics, build_predictions_report, evaluate_predictions,
    make_wrmsse_metric,
)
from src.modeling import (
    backward_feature_selection, catboost_features, compute_permutation_importance,
    drift, fit_catboost, fit_ets, fit_histgb, fit_lightgbm, fit_prophet, fit_ridge,
    fit_sarima, fit_tbats, fit_theta, fit_xgboost, histgb_features, historical_mean,
    moving_average, naive_last_value, seasonal_naive,
)


# ============================== CONFIG ==============================
# Todo lo que controla la corrida vive acá, agrupado por fase. Editar y correr.

@dataclass
class Config:
    # --- selección de dataset ---
    level_id: int = 12
    target: str = "sales"  # "sales" o "cumN" (ver config.CUM_HORIZONS)

    # --- fases on/off ---
    run_baseline_naive: bool = True
    run_baseline_stats: bool = False    # SARIMA/ETS/Theta/TBATS/Prophet: serie x serie, lento
    run_baseline_ml: bool = False       # LightGBM/XGBoost/CatBoost/HistGB/Ridge, hiperparámetros default
    run_feature_selection: bool = False  # permutation importance + backward elimination
    run_shap: bool = False
    run_optuna: bool = False
    save_artifact: bool = True

    # --- split temporal ---
    # daily: cantidad de días. weekly: cantidad de filas semanales (se convierte a
    # días x7 antes de llamar a date_split, que siempre trabaja en días de calendario).
    valid_periods_daily: int = 365
    test_periods_daily: int = 28
    valid_periods_weekly: int = 52
    test_periods_weekly: int = 4

    # --- comparación de modelos base ---
    ma_windows: dict = field(default_factory=lambda: {
        "daily": [7, 14, 21, 28, 35], "weekly": [2, 3, 4, 6, 8],
    })
    random_state: int = 42

    # --- selección de features ---
    permutation_sample_size: int = 15_000
    permutation_n_repeats: int = 3
    backward_tolerance: float = 0.001

    # --- SHAP ---
    shap_sample_size: int = 2_000

    # --- Optuna ---
    optuna_n_trials: int = 1_000
    optuna_timeout_s: int = 15 * 60

    # --- modelo final ---
    final_n_estimators: int = 1_500
    # "tweedie" o "regression_l2"/"rmse" (aplica tanto si corre Optuna como si no).
    objective: str = "tweedie"
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


# ============================== FASES ==============================

def load_data(cfg: Config) -> SimpleNamespace:
    level = config.LEVELS_BY_ID[cfg.level_id]
    grain = level.grains[0]
    level_str = f"level_{level.id:02d}_{grain}_{level.name}"

    df = pd.read_parquet(config.featured_level_path(level, grain)).sort_values("date").reset_index(drop=True)

    target = cfg.target
    id_cols = ["agg_id", "date"]
    cum_cols = [c for c in df.columns if c.startswith("cum") and c[3:].isdigit()]
    leaky_cols = [c for c in (["gross_sales"] + cum_cols) if c != target]

    features = [c for c in df.columns if c not in id_cols + leaky_cols + [target]]
    categorical_features = [
        c for c in [
            "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        ] if c in df.columns
    ]
    numerical_features = [c for c in features if c not in categorical_features]
    features = categorical_features + numerical_features

    logger.info("{} | target={} | {:,} filas | {} features ({} categóricas)",
                level_str, target, len(df), len(features), len(categorical_features))

    return SimpleNamespace(
        level=level, grain=grain, level_str=level_str, df=df, target=target,
        id_cols=id_cols, leaky_cols=leaky_cols,
        features=features, categorical_features=categorical_features,
        numerical_features=numerical_features,
        model_results=[], predictions_valid={},
    )


def split_data(state: SimpleNamespace, cfg: Config) -> None:
    if state.grain == "daily":
        valid_days, test_days = cfg.valid_periods_daily, cfg.test_periods_daily
    else:
        # date_split trabaja en días de calendario: convertir semanas -> días. El
        # -1 interno de date_split alinea el corte para que siga cayendo exactamente
        # en un múltiplo de 7 filas semanales (no corta una semana a la mitad).
        valid_days, test_days = cfg.valid_periods_weekly * 7, cfg.test_periods_weekly * 7

    split = date_split(state.df, valid_days=valid_days, test_days=test_days)
    split.log_summary()

    X_train, y_train, X_valid, y_valid, X_test, y_test = build_feature_matrices(
        state.df, split, state.features, state.categorical_features, state.target,
    )
    state.X_train, state.y_train = X_train, y_train
    state.X_valid, state.y_valid = X_valid, y_valid
    state.X_test, state.y_test = X_test, y_test

    # train/valid/test solo se usan después para bookkeeping de evaluación (scales
    # WRMSSE/MASE, pesos por precio, clip de cierres, gross_sales de reportes) --
    # no las ~100 columnas de features (esas ya están en X_train/X_valid/X_test).
    # Cargar el nivel 12 completo (mayor cardinalidad) puede acercarse al límite de
    # RAM disponible; recortar acá evita cargar ese peso tres veces más.
    eval_cols = [c for c in dict.fromkeys(
        ["agg_id", "date", "sales", "gross_sales", "avg_sell_price", "is_store_closed", state.target]
    ) if c in state.df.columns]
    state.train = split.train[eval_cols].copy()
    state.valid = split.valid[eval_cols].copy()
    state.test = split.test[eval_cols].copy()
    state.first_date, state.valid_start, state.test_start = (
        split.first_date, split.valid_start, split.test_start,
    )
    del split, state.df
    gc.collect()

    state.wrmsse_metric = make_wrmsse_metric(state.train, state.valid)

    logger.info("Features: {} ({} categóricas)", len(state.features), len(state.categorical_features))


def make_evaluator(state: SimpleNamespace):
    """Cierra sobre `state` para acumular cada modelo evaluado en
    `model_results`/`predictions_valid`, igual que `evaluate_model` en el notebook."""
    def evaluate_model(name, y_pred_valid, fit_time=None, category=None):
        result = evaluate_predictions(
            state.train, state.valid, state.y_valid, y_pred_valid, name,
            fit_time=fit_time, category=category,
        )
        state.model_results.append(result)
        state.predictions_valid[name] = y_pred_valid
        return result

    return evaluate_model


def run_baseline_naive(state: SimpleNamespace, cfg: Config, evaluate_model) -> None:
    season_length = {"daily": 7, "weekly": 52}[state.grain]
    grain_letter = state.grain[0]

    t0 = time.perf_counter()
    evaluate_model("Naive (last value)", naive_last_value(state.train, state.valid, state.target),
                    fit_time=time.perf_counter() - t0, category="Naive")

    t0 = time.perf_counter()
    y_pred = seasonal_naive(state.train, state.valid, state.target, season_length=season_length)
    evaluate_model(f"Seasonal naive ({season_length}{grain_letter})", y_pred,
                    fit_time=time.perf_counter() - t0, category="Naive")

    if state.grain == "daily":
        t0 = time.perf_counter()
        y_pred = seasonal_naive(state.train, state.valid, state.target, season_length=365)
        evaluate_model("Seasonal naive (365d)", y_pred, fit_time=time.perf_counter() - t0, category="Naive")

    t0 = time.perf_counter()
    evaluate_model("Drift", drift(state.train, state.valid, state.target),
                    fit_time=time.perf_counter() - t0, category="Naive")

    t0 = time.perf_counter()
    evaluate_model("Historical mean", historical_mean(state.train, state.valid, state.target),
                    fit_time=time.perf_counter() - t0, category="Naive")

    for window in cfg.ma_windows[state.grain]:
        t0 = time.perf_counter()
        y_pred = moving_average(state.train, state.valid, state.target, window=window)
        evaluate_model(f"Moving average ({window}{grain_letter})", y_pred,
                        fit_time=time.perf_counter() - t0, category="Naive")


def run_baseline_stats(state: SimpleNamespace, evaluate_model) -> None:
    season_length = {"daily": 7, "weekly": 52}[state.grain]

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
                               random_state=cfg.random_state)
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
    out_dir = config.ARTIFACTS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(state.model_results).sort_values("wrmsse")
    results_df.to_csv(out_dir / f"{state.level_str}_{state.target}_model_comparison.csv", index=False)

    if state.predictions_valid:
        series_metrics_df = build_all_series_metrics(
            state.train, state.valid, state.y_valid, state.predictions_valid, target_col=state.target,
        )
        series_metrics_df.to_csv(out_dir / f"{state.level_str}_{state.target}_series_metrics.csv", index=False)

    logger.info("Comparación de modelos guardada en {}", out_dir)


def select_features(state: SimpleNamespace, cfg: Config) -> None:
    lgbm_model = getattr(state, "lgbm_model", None)
    if lgbm_model is None:
        lgbm_model = fit_lightgbm(state.X_train, state.y_train, state.categorical_features,
                                   random_state=cfg.random_state)

    importance_gain = pd.Series(
        lgbm_model.booster_.feature_importance(importance_type="gain"), index=state.features,
    ).sort_values(ascending=False)
    logger.info("Top 10 gain importance:\n{}", importance_gain.head(10))

    importance_perm = compute_permutation_importance(
        lgbm_model, state.X_valid, state.y_valid, state.features,
        sample_size=cfg.permutation_sample_size, n_repeats=cfg.permutation_n_repeats,
        random_state=cfg.random_state,
    )
    logger.info("Top 10 permutation importance:\n{}", importance_perm.head(10))

    selected_features, final_wrmsse, selection_log = backward_feature_selection(
        state.X_train, state.y_train, state.X_valid, state.train, state.valid,
        state.features, state.categorical_features, importance_perm,
        tolerance=cfg.backward_tolerance, random_state=cfg.random_state,
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

    logger.info("Features finales: {} ({} categóricas) | WRMSSE={:.4f}",
                len(state.features), len(state.categorical_features), final_wrmsse)


def run_shap(state: SimpleNamespace, cfg: Config, evaluate_model) -> None:
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import shap
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(objective="rmse")
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

    objective_kwargs = resolve_objective(cfg)

    def objective(trial):
        model = lgb.LGBMRegressor(
            **objective_kwargs,
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            n_estimators=cfg.final_n_estimators,
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
        model.fit(
            state.X_train, state.y_train,
            eval_set=[(state.X_valid, state.y_valid)],
            eval_metric="rmse",
            categorical_feature=state.categorical_features,
            callbacks=[
                lgb.early_stopping(30, first_metric_only=True, verbose=False),
                LightGBMPruningCallback(trial, "rmse"),
            ],
        )
        y_pred = model.predict(state.X_valid, num_iteration=model.best_iteration_)
        _, final_wrmsse, _ = state.wrmsse_metric(state.y_valid, y_pred)
        return final_wrmsse

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

    logger.info("Mejor WRMSSE (Optuna): {:.4f}", study.best_trial.value)
    logger.info("Mejores hiperparámetros: {}", study.best_trial.params)
    return study.best_trial.params


def fit_final_model(state: SimpleNamespace, cfg: Config, evaluate_model, best_params: dict) -> None:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    objective_kwargs = resolve_objective(cfg)

    if best_params:
        model_params = dict(best_params)
        final_model = LGBMRegressor(
            **objective_kwargs,
            n_estimators=cfg.final_n_estimators,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
            subsample_freq=1,
            **model_params,
        )
        label = "LightGBM (final, Optuna)"
    else:
        model_params = {**cfg.default_lgbm_params, "seed": cfg.random_state, **objective_kwargs}
        final_model = LGBMRegressor(**model_params)
        label = "LightGBM (final, params default)"

    t0 = time.perf_counter()
    final_model.fit(
        state.X_train, state.y_train,
        eval_set=[(state.X_valid, state.y_valid)],
        eval_metric=state.wrmsse_metric,
        categorical_feature=state.categorical_features,
        callbacks=[lgb.early_stopping(50, first_metric_only=True), lgb.log_evaluation(10)],
    )
    fit_time = time.perf_counter() - t0
    evaluate_model(label, final_model.predict(state.X_valid), fit_time=fit_time, category="ML")

    metrics_test, df_pred = build_predictions_report(
        state.train, state.test, state.y_test, final_model.predict(state.X_test), target_col=state.target,
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

    # No se guardan df/X_train/y_train/X_valid/y_valid/valid: reconstruibles desde
    # el parquet en data/datasets/ + este mismo pipeline, y notebooks/03_predictions.ipynb
    # no los usa -- guardarlos solo triplicaba el tamaño del artifact (y el pico de
    # RAM al armarlo) sin necesidad, justo lo que hace fallar por memoria al nivel 12.
    artifact = {
        "level": state.level_str,
        "model": state.final_model,
        "model_params": state.model_params,
        "X_test": state.X_test, "y_test": state.y_test,
        "train": state.train, "test": state.test,
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

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = config.MODELS_DIR / f"{state.level_str}_{state.target}_artifact.pkl"
    joblib.dump(artifact, artifact_path)
    logger.success("Artifact guardado en {}", artifact_path)


def main():
    t_start = time.perf_counter()
    cfg = CFG

    state = load_data(cfg)
    split_data(state, cfg)
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


if __name__ == "__main__":
    main()
