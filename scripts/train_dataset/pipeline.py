import time
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

from src.data.dataset import load_data, split_data
from src.evaluation import evaluate_predictions, objective_metric
from src.evaluation.scaled import clip_closed_stores
from src.modeling import ALL_FAMILIES
from scripts.train_dataset.config import Config
from scripts.train_dataset.benchmarking import (
    pick_winner, run_baseline_naive, run_baseline_stats, run_bench_ml, save_model_comparison,
)
from scripts.train_dataset.refinement import fit_final_model, run_shap, select_features, tune_optuna
from scripts.train_dataset.export import export_artifact


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


def run_pipeline(cfg: Config, dataset_path: Path | None = None) -> None:
    t_start = time.perf_counter()

    state = load_data(cfg.level_id, "sales", dataset_path, grain=cfg.grain)
    split_data(state, "sales")
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
        logger.info("Fase bench ML ({} familias, sin Optuna): {:.1f}s",
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
