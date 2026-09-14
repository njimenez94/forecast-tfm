"""Recalcula las métricas de test (wape_test/wrmsse_test/spec_test/...) de artifacts
ya exportados, sin reentrenar: reusa el `model` ya fitteado, reconstruye
X_test/y_test/train/test desde el parquet (misma lógica que
notebooks/04_hierarchical_comparison.ipynb) y recalcula con build_predictions_report.

Pensado para dos casos:
  - Backfill de artifacts viejos que no tenían todas las métricas de test (p.ej.
    export_artifact solo guardaba wape_test/wrmsse_test antes de este script).
  - Recalcular tras cambios en el parquet de datasets o en build_predictions_report,
    sin pagar el costo de reentrenar el modelo.

Por default salta los artifacts que ya tienen todas las claves `*_test` que produce
build_predictions_report (idempotente, rápido en reruns); `--force` recalcula todos.

Uso:
    python -m scripts.backfill_test_metrics
    python -m scripts.backfill_test_metrics --force
    python -m scripts.backfill_test_metrics --level 9
"""
import argparse

import joblib
from loguru import logger

import config
from src.data.dataset import reconstruct_test_data
from src.evaluation import build_predictions_report


def backfill(path, force: bool) -> bool:
    """Devuelve True si reescribió el artifact en `path`."""
    artifact = joblib.load(path)

    X_test, y_test, train, test = reconstruct_test_data(artifact)
    y_pred = artifact["model"].predict(X_test)
    metrics, _ = build_predictions_report(train, test, y_test, y_pred, target_col=artifact["target"])
    new_fields = {f"{k}_test": v for k, v in metrics.items()}

    if not force and all(artifact.get(k) == v for k, v in new_fields.items()):
        logger.info("[skip] {} ya tiene estas métricas de test", path.name)
        return False

    artifact.update(new_fields)
    joblib.dump(artifact, path)
    logger.success(
        "[ok] {} | wape_test={:.2%} wrmsse_test={:.4f} spec_test={:.2f}",
        path.name, metrics["wape"], metrics["wrmsse"], metrics["spec"],
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, default=None, help="Solo backfillear este level_id (default: todos)")
    parser.add_argument("--force", action="store_true", help="Recalcular aunque ya tenga las métricas")
    args = parser.parse_args()

    pattern = f"level_{args.level:02d}_*_artifact.pkl" if args.level else "*_artifact.pkl"
    paths = sorted(config.MODELS_DIR.glob(pattern))
    logger.info("{} artifacts encontrados ({})", len(paths), pattern)

    updated = sum(backfill(path, args.force) for path in paths)
    logger.info("{}/{} artifacts actualizados", updated, len(paths))


if __name__ == "__main__":
    main()
