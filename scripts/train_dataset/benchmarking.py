import time
from types import SimpleNamespace

import pandas as pd
from loguru import logger

import config
from src.evaluation import build_all_series_metrics
from src.modeling import (
    ALL_FAMILIES, MODEL_FAMILIES, TREE_FAMILIES,
    drift, fit_ets, fit_prophet, fit_sarima, fit_tbats, fit_theta, historical_mean,
    moving_average, seasonal_naive,
)
from scripts.train_dataset.config import Config


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


def run_bench_ml(state: SimpleNamespace, cfg: Config, evaluate_model) -> None:
    """Compara las `ALL_FAMILIES` con hiperparámetros default (fijos por familia) +
    early stopping -- sin Optuna acá, para no gastar tiempo tuneando 5 familias
    cuando solo una sigue adelante. Optuna corre una única vez, más tarde, sobre la
    familia ganadora (ver refinement.tune_optuna). `pick_winner` decide el ganador
    entre estos resultados (solo familias tree-based, ver TREE_FAMILIES)."""
    state.bench_models = {}
    state.bench_params = {}
    state.bench_scores = {}

    for i, family_name in enumerate(ALL_FAMILIES, 1):
        logger.info("Bench ML: familia {}/{} -- {}", i, len(ALL_FAMILIES), family_name)
        family = MODEL_FAMILIES[family_name]
        params = family.resolve_objective(cfg.objective, cfg.tweedie_variance_power)
        if family.n_estimators_param:
            params = {**params, family.n_estimators_param: cfg.bench_n_estimators}

        t0 = time.perf_counter()
        fitted = family.fit(
            state.X_train, state.y_train, state.X_valid, state.y_valid,
            state.categorical_features, state.numerical_features,
            params, early_stopping_rounds=30, random_state=cfg.random_state,
        )
        fit_time = time.perf_counter() - t0
        result = evaluate_model(f"{family_name} (bench)", fitted.predict(state.X_valid),
                                 fit_time=fit_time, category="ML", stage="bench")

        state.bench_models[family_name] = fitted
        state.bench_params[family_name] = params
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
    out_dir = config.ARTIFACTS_DIR / "results"
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
