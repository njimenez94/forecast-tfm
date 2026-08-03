"""Entrena un MLForecast por nivel de agregación y evalúa en valid y test.

El final de cada serie se reserva en dos bloques consecutivos de igual longitud
(~12 meses, max(HORIZON) del grain): valid (pensado para tuning, p.ej. Optuna, o
early stopping) y test (prueba final, nunca usado para entrenar). El modelo se
ajusta solo con train. El tamaño del bloque es el mismo para todos los targets,
independiente de la ventana N de los targets acumulados, para que valid/test sean
comparables entre targets.

Para 'sales' evalúa en múltiples sub-horizontes (HORIZON config) dentro del bloque.
Para 'cumN' evalúa un único horizonte igual al bloque completo: como "y" ya es la
suma móvil de los N períodos siguientes a cada fecha, comparar período a período a
lo largo de todo el bloque agrega esa suma móvil de forma consistente en el tiempo.

Los modelos se guardan en artifacts/models/ y las métricas en artifacts/results.parquet.

Uso:
    python -m scripts.train_datasets
"""
import gc
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Found null values")
# Momentum (RollingMean corta / larga) da 0/0 en series intermitentes con ventana
# en cero; el NaN resultante es válido para LightGBM, no un error.
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

import pandas as pd
from loguru import logger

import config
from src.data.split import LevelSplit, parse_level_file, prepare_level
from src.evaluation.metrics import bias, compute_wrmsse, wape
from src.modeling.train import build_fcst, encode_static


def _tfm_window(t) -> int:
    """Ventana de historia que necesita un lag_transform (0 si no tiene una fija,
    p.ej. ExpandingMean; recursivo para Combine, que envuelve otros dos transforms)."""
    if hasattr(t, "window_size"):
        return t.window_size
    if hasattr(t, "tfm1"):
        return max(_tfm_window(t.tfm1), _tfm_window(t.tfm2))
    return 0


def _eval_horizon(preds_df: pd.DataFrame, valid_pd: pd.DataFrame,
                  train_for_wrmsse: pd.DataFrame | None, h: int) -> dict:
    """Compara, período a período, las primeras h filas de preds_df contra valid_pd.

    Para targets acumulados 'cumN', "y" ya es la suma de los N períodos siguientes a
    cada fecha, así que esto evalúa esa suma móvil de forma consistente a lo largo de
    todo el bloque (h = tamaño del bloque); WRMSSE no aplica y se omite (train_for_wrmsse=None).
    """
    mask_p = preds_df.groupby("unique_id", sort=False).cumcount() < h
    mask_v = valid_pd.groupby("unique_id", sort=False).cumcount() < h

    preds   = preds_df[mask_p]["lgb"].clip(lower=0).round(0).astype(int).to_numpy()
    actuals = valid_pd[mask_v]["y"].to_numpy()

    wrmsse = float("nan")
    if train_for_wrmsse is not None:
        valid_h = (valid_pd[mask_v]
                   .rename(columns={"unique_id": "agg_id", "ds": "date", "y": "sales"}))
        wrmsse = compute_wrmsse(train_for_wrmsse, valid_h, preds)

    return {
        "wape":   float(wape(actuals, preds)),
        "bias":   float(bias(actuals, preds)),
        "wrmsse": wrmsse,
    }


def _split_preds(preds_df: pd.DataFrame, block: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa las predicciones en bloque valid (primeros `block` pasos) y test (siguientes)."""
    cumcount = preds_df.groupby("unique_id", sort=False).cumcount()
    preds_valid = preds_df[cumcount < block].reset_index(drop=True)
    preds_test  = preds_df[cumcount >= block].reset_index(drop=True)
    return preds_valid, preds_test


def _train_level(file: Path, level, grain: str, target: str) -> list[dict]:
    t0 = time.perf_counter()
    n = config.cum_n(target)

    # Bloque fijo (~12 meses) para valid/test, igual para todos los targets: el tamaño
    # de la ventana acumulada (7, 14, ...) no debe achicar el período de evaluación.
    block     = max(config.HORIZON[grain])
    h_predict = 2 * block
    horizons  = config.HORIZON[grain] if n is None else [block]

    data: LevelSplit = prepare_level(file, level, grain, block, target)
    train_start  = data.train_pd["ds"].min().date()
    train_end    = data.train_pd["ds"].max().date()
    n_train_rows = len(data.train_pd)
    rows_per_series = n_train_rows // max(data.n_series, 1)

    lags = config.valid_lags(grain, target)
    transforms = config.valid_lag_transforms(grain, target)
    transform_extra = max(
        (k + max((_tfm_window(t) for t in ts), default=0) for k, ts in transforms.items()),
        default=0,
    )
    min_needed = max(max(lags), transform_extra)

    logger.info(
        "L{} {}/{} | {} | {} series | bloque={} | train={:,} [{} → {}] valid={:,} test={:,}",
        level.id, level.name, grain, target, data.n_series,
        block, n_train_rows, train_start, train_end, len(data.valid_pd), len(data.test_pd),
    )
    if rows_per_series <= min_needed + 1:
        logger.warning(
            "  SKIP: historia insuficiente ({} períodos/serie, mínimo {})",
            rows_per_series, min_needed + 2,
        )
        return []

    encode_static(data.train_pd, data.static_cols)
    fcst = build_fcst(grain, target)
    t_fit = time.perf_counter()
    fcst.fit(data.train_pd, static_features=data.static_cols)
    fit_time_s = time.perf_counter() - t_fit
    del data.train_pd; gc.collect()

    # future_exog cubre valid+test; se trunca a los h_predict pasos que hacen falta
    X_df = None
    if data.avail_exog and data.future_exog is not None:
        X_df = (
            data.future_exog
            .sort_values("ds")
            .groupby("unique_id", sort=False)
            .head(h_predict)
            .reset_index(drop=True)
        )

    preds_df = fcst.predict(h=h_predict, X_df=X_df)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"mlf_level_{level.id:02d}_{grain}_{target}_{level.name}"
    fcst.save(str(model_path))
    del fcst; gc.collect()

    preds_valid, preds_test = _split_preds(preds_df, block)

    results = []
    for split_name, target_pd, preds_split in (
        ("valid", data.valid_pd, preds_valid),
        ("test", data.test_pd, preds_test),
    ):
        for h in horizons:
            m = _eval_horizon(preds_split, target_pd, data.train_for_wrmsse, h)

            logger.debug(  # detalle por horizonte visible solo en nivel DEBUG
                "  {:>5} h={:>3} | WAPE: {:.1%}  BIAS: {:+.1%}  WRMSSE: {}",
                split_name, h, m["wape"], m["bias"],
                f"{m['wrmsse']:.3f}" if m["wrmsse"] == m["wrmsse"] else "n/a",
            )
            results.append({
                "level_id": level.id, "level_name": level.name, "grain": grain,
                "target": target,
                "target_type": "sales" if n is None else "cumulative",
                "split": split_name,
                "horizon": h,
                "n_series": data.n_series,
                "n_rows": n_train_rows + len(data.valid_pd) + len(data.test_pd),
                "fit_time_s": round(fit_time_s, 2),
                **m,
                "model_path": str(model_path),
            })

    logger.debug("L{} {}/{}/{} listo en {:.1f}s", level.id, level.name, grain, target,
                 time.perf_counter() - t0)
    return results


def _print_group_summary(group_rows: list[dict]) -> None:
    """Imprime tabla compacta con todos los resultados de un nivel/grain (todos los targets)."""
    if not group_rows:
        return
    r0 = group_rows[0]
    header = f"── L{r0['level_id']} {r0['level_name']} / {r0['grain']} ({r0['n_series']} series) ──"
    logger.info("{}", "─" * max(len(header), 60))
    logger.info("{}", header)
    logger.info("  {:>8}  {:>5}  {:>5}  {:>7}  {:>7}  {:>8}", "target", "split", "h", "WAPE", "BIAS", "WRMSSE")
    for row in group_rows:
        wrmsse = f"{row['wrmsse']:.3f}" if row["wrmsse"] == row["wrmsse"] else "  n/a"
        logger.info(
            "  {:>8}  {:>5}  {:>5}  {:>6.1%}  {:>+7.1%}  {:>8}",
            row["target"], row["split"], row["horizon"], row["wape"], row["bias"], wrmsse,
        )


def main():
    files = sorted(config.FEATURED_DIR.glob("dataset_level_*.parquet"))
    if not files:
        logger.error("No hay parquets en {}. Ejecuta primero: make process-data && make build-datasets",
                     config.FEATURED_DIR)
        return

    results = []

    for i, file in enumerate(files, 1):
        parsed = parse_level_file(file)
        if parsed is None:
            logger.warning("  SKIP legacy: {}", file.name)
            continue
        level, grain = parsed

        targets = ["sales"] + [f"cum{n}" for n in config.CUM_HORIZONS[grain]]
        group_rows: list[dict] = []
        for target in targets:
            try:
                group_rows.extend(_train_level(file, level, grain, target))
            except Exception as e:
                logger.exception("[{}/{}] error en {} target={}: {}", i, len(files), file.name, target, e)
        results.extend(group_rows)
        _print_group_summary(group_rows)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(config.OUTPUT_DIR / "experiment_results.parquet")
    logger.info("─" * 60)
    logger.info("Resultados guardados en {}", config.OUTPUT_DIR / "experiment_results.parquet")


if __name__ == "__main__":
    main()
