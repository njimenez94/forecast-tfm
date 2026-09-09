"""Genera artifacts/test_sets/{level}.parquet: predicción del modelo final de cada
nivel sobre su test set, con columnas descriptivas (dims + fecha + precio) en vez de
las ~180 features de entrenamiento -- pensado para análisis exploratorio de errores
por nivel/serie/fecha (ver notebooks/03_predictions.ipynb para el detalle por artifact
individual). También exporta output/test_metrics.json con las mismas métricas de
error que se comparan en artifacts/results/*_model_comparison.csv (evaluate_predictions),
pero del modelo final sobre test, agregadas a nivel de... nivel (todas las combinaciones
dept/store de un nivel juntas).

Un parquet por nivel (12): niveles con split_by (10-12) concatenan todas sus
combinaciones dept/store en un único parquet.

    make test-sets
    # o
    uv run python -m scripts.build_test_sets
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger

import config
from src.data.dataset import reconstruct_test_data
from src.logging_setup import configure_logging
from src.evaluation import evaluate_predictions
from src.evaluation.scaled import clip_closed_stores

DESCRIPTIVE_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
OUT_DIR = config.ARTIFACTS_DIR / "test_sets"
METRICS_PATH = config.OUTPUT_DIR / "test_metrics.json"


def predict_artifact(artifact_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (test, train) del artifact: test con dims + forecast (para el
    parquet), train para las escalas por serie de wrmsse/mase/spec (evaluate_predictions)."""
    artifact = joblib.load(artifact_path)
    X_test, _, train, test = reconstruct_test_data(artifact)

    dim_cols = [c for c in DESCRIPTIVE_COLS if c in X_test.columns]
    test = pd.concat([test.reset_index(drop=True), X_test[dim_cols].reset_index(drop=True)], axis=1)
    test["forecast"] = clip_closed_stores(test, artifact["model"].predict(X_test)).round(0)
    return test, train


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for level in config.LEVELS:
        grain = level.grains[0]
        artifact_paths = sorted(config.MODELS_DIR.glob(f"level_{level.id:02d}_*_sales_artifact.pkl"))
        if not artifact_paths:
            logger.warning("Nivel {}: sin artifacts en {}", level.id, config.MODELS_DIR)
            continue

        test_frames, train_frames = [], []
        for path in artifact_paths:
            try:
                test, train = predict_artifact(path)
                test_frames.append(test)
                train_frames.append(train)
            except Exception:
                logger.exception("Nivel {}: falló {}, sigue con el resto.", level.id, path.name)

        if not test_frames:
            continue

        df = pd.concat(test_frames, ignore_index=True)
        level_str = f"level_{level.id:02d}_{grain}_{level.name}"
        out_path = OUT_DIR / f"{level_str}_test.parquet"
        df.to_parquet(out_path, index=False)
        logger.success("Nivel {}: {:,} filas ({} artifact(s)) -> {}",
                        level.id, len(df), len(artifact_paths), out_path)

        train_df = pd.concat(train_frames, ignore_index=True)
        metrics = evaluate_predictions(train_df, df, df["sales"], df["forecast"], name=level_str)
        metrics["level_id"] = level.id
        metrics["n_series"] = df["series_id"].nunique()
        metrics["n_rows"] = len(df)
        all_metrics.append(metrics)

    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2, default=float))
    logger.success("Métricas de {} niveles -> {}", len(all_metrics), METRICS_PATH)


if __name__ == "__main__":
    configure_logging("build_test_sets")
    main()
