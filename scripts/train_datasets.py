"""Entrena un MLForecast por nivel de agregación y evalúa en los últimos H períodos.

Para el target 'sales' evalúa en múltiples sub-horizontes (HORIZON config).
Para targets 'cumN' predice h=1 (el acumulado de N períodos) y lo compara contra el
primer paso del conjunto de validación, que ya contiene el acumulado real pre-computado.

Los modelos se guardan en artifacts/models/ y las métricas en artifacts/results.parquet.

Uso:
    python -m scripts.train_datasets
"""
import gc
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Found null values")

import pandas as pd
from loguru import logger

import config
from src.data.split import LevelSplit, parse_level_file, prepare_level
from src.evaluation.metrics import bias, compute_wrmsse, wape
from src.modeling.train import build_fcst, encode_static


def _eval_horizon(preds_df: pd.DataFrame, valid_pd: pd.DataFrame,
                  train_for_wrmsse: pd.DataFrame, h: int) -> dict:
    mask_p = preds_df.groupby("unique_id", sort=False).cumcount() < h
    mask_v = valid_pd.groupby("unique_id", sort=False).cumcount() < h

    preds   = preds_df[mask_p]["lgb"].clip(lower=0).round(0).astype(int).to_numpy()
    actuals = valid_pd[mask_v]["y"].to_numpy()
    valid_h = (valid_pd[mask_v]
               .rename(columns={"unique_id": "agg_id", "ds": "date", "y": "sales"}))

    return {
        "wape":   float(wape(actuals, preds)),
        "bias":   float(bias(actuals, preds)),
        "wrmsse": compute_wrmsse(train_for_wrmsse, valid_h, preds),
    }


def _eval_cum(preds_df: pd.DataFrame, valid_pd: pd.DataFrame) -> dict:
    """Evalúa un modelo de target acumulado (h=1).

    Compara la única predicción por serie contra el primer paso del conjunto de
    validación, que contiene la suma acumulada real del período siguiente.
    """
    first_valid = (
        valid_pd.sort_values("ds")
        .groupby("unique_id", sort=False)
        .first()
        .reset_index()[["unique_id", "y"]]
    )
    merged  = preds_df[["unique_id", "lgb"]].merge(first_valid, on="unique_id", how="inner")
    preds   = merged["lgb"].clip(lower=0).to_numpy()
    actuals = merged["y"].to_numpy()
    return {
        "wape":   float(wape(actuals, preds)),
        "bias":   float(bias(actuals, preds)),
        "wrmsse": float("nan"),  # no aplica para targets acumulados
    }


def _train_level(file: Path) -> list[dict]:
    t0 = time.perf_counter()
    parsed = parse_level_file(file)
    if parsed is None:
        logger.warning("  SKIP: formato no reconocido (¿archivo legacy sin target?): {}", file.name)
        return []
    level, grain, wlabel, target = parsed
    n = config.cum_n(target)

    if n is None:
        # Target 'sales': evaluar en múltiples horizontes
        horizons     = config.HORIZON[grain]
        h_max        = max(horizons)
        split_horizon = h_max
        h_predict    = h_max
    else:
        # Target acumulado 'cumN': reservar N períodos para validación, predecir h=1
        horizons      = [n]
        h_max         = n
        split_horizon = n
        h_predict     = 1

    data: LevelSplit = prepare_level(file, level, grain, split_horizon, target)
    train_start  = data.train_pd["ds"].min().date()
    train_end    = data.train_pd["ds"].max().date()
    n_train_rows = len(data.train_pd)
    rows_per_series = n_train_rows // max(data.n_series, 1)

    lags = config.valid_lags(grain, target)
    transforms = config.valid_lag_transforms(grain, target)
    transform_extra = max(
        (k + max(t.window_size for t in ts) for k, ts in transforms.items()),
        default=0,
    )
    min_needed = max(max(lags), transform_extra)

    logger.info(
        "L{} {}/{} | {} | {} series | h={} | train={:,} [{}  {} → {}] valid={:,}",
        level.id, level.name, grain, target, data.n_series,
        h_max, n_train_rows, wlabel, train_start, train_end, len(data.valid_pd),
    )
    if rows_per_series <= min_needed + 1:
        logger.warning(
            "  SKIP: ventana {} insuficiente ({} períodos/serie, mínimo {})",
            wlabel, rows_per_series, min_needed + 2,
        )
        return []

    encode_static(data.train_pd, data.static_cols)
    fcst = build_fcst(grain, target)
    t_fit = time.perf_counter()
    fcst.fit(data.train_pd, static_features=data.static_cols)
    fit_time_s = time.perf_counter() - t_fit
    del data.train_pd; gc.collect()

    # Para h=1 (targets acumulados) truncar future_exog a 1 fila por serie
    X_df = None
    if data.avail_exog and data.future_exog is not None:
        if h_predict == 1:
            X_df = (
                data.future_exog
                .sort_values("ds")
                .groupby("unique_id", sort=False)
                .head(1)
                .reset_index(drop=True)
            )
        else:
            X_df = data.future_exog

    preds_df = fcst.predict(h=h_predict, X_df=X_df)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"mlf_level_{level.id:02d}_{grain}_{wlabel}_{target}_{level.name}"
    fcst.save(str(model_path))
    del fcst; gc.collect()

    results = []
    for h in horizons:
        if n is None:
            m = _eval_horizon(preds_df, data.valid_pd, data.train_for_wrmsse, h)
        else:
            m = _eval_cum(preds_df, data.valid_pd)

        logger.debug(  # detalle por horizonte visible solo en nivel DEBUG
            "  h={:>3} | WAPE: {:.1%}  BIAS: {:+.1%}  WRMSSE: {}",
            h, m["wape"], m["bias"],
            f"{m['wrmsse']:.3f}" if m["wrmsse"] == m["wrmsse"] else "n/a",
        )
        results.append({
            "level_id": level.id, "level_name": level.name, "grain": grain,
            "target": target,
            "target_type": "sales" if n is None else "cumulative",
            "window": wlabel, "horizon": h,
            "n_series": data.n_series,
            "n_rows": n_train_rows + len(data.valid_pd),
            "fit_time_s": round(fit_time_s, 2),
            **m,
            "model_path": str(model_path),
        })

    logger.debug("L{} {}/{}/{} listo en {:.1f}s", level.id, level.name, grain, target,
                 time.perf_counter() - t0)
    return results


def _sort_key(f: Path):
    """Ordena parquets por (level_id, grain, window, target_n) para log legible.

    Sin esto glob devuelve orden alfabético: cum14 < cum21 < cum365 < cum42 < cum7.
    Con esto: sales=0, cum7, cum14, cum21, ..., cum365.
    """
    parsed = parse_level_file(f)
    if parsed is None:
        return (999, "", "", 9999)
    level, grain, wlabel, target = parsed
    n = config.cum_n(target) or 0
    grain_ord = 0 if grain == "daily" else 1
    window_ord = {"w1y": 0, "w2y": 1, "w3y": 2, "w4y": 3, "wmax": 4}.get(wlabel, 99)
    return (level.id, grain_ord, window_ord, n)


def _print_group_summary(group_rows: list[dict]) -> None:
    """Imprime tabla compacta con todos los resultados de un grupo (level/grain/window)."""
    if not group_rows:
        return
    r0 = group_rows[0]
    header = (f"── L{r0['level_id']} {r0['level_name']} / {r0['grain']} / "
              f"{r0['window']} ({r0['n_series']} series) ──")
    logger.info("{}", "─" * max(len(header), 60))
    logger.info("{}", header)
    logger.info("  {:>8}  {:>5}  {:>7}  {:>7}  {:>8}", "target", "h", "WAPE", "BIAS", "WRMSSE")
    for row in group_rows:
        wrmsse = f"{row['wrmsse']:.3f}" if row["wrmsse"] == row["wrmsse"] else "  n/a"
        logger.info(
            "  {:>8}  {:>5}  {:>6.1%}  {:>+7.1%}  {:>8}",
            row["target"], row["horizon"], row["wape"], row["bias"], wrmsse,
        )


def main():
    files = sorted(config.PROCESSED_DIR.glob("dataset_level_*.parquet"), key=_sort_key)
    if not files:
        logger.error("No hay parquets en {}. Ejecuta primero: make create_datasets",
                     config.PROCESSED_DIR)
        return

    results = []
    group_rows: list[dict] = []
    current_group: tuple = ()

    for i, file in enumerate(files, 1):
        parsed = parse_level_file(file)
        if parsed is None:
            logger.warning("  SKIP legacy: {}", file.name)
            continue

        level, grain, wlabel, _ = parsed
        group_key = (level.id, grain, wlabel)

        # Al cambiar de grupo, imprimir tabla del grupo anterior
        if group_key != current_group and group_rows:
            _print_group_summary(group_rows)
            group_rows = []
        current_group = group_key

        try:
            rows = _train_level(file)
            results.extend(rows)
            group_rows.extend(rows)
        except Exception as e:
            logger.exception("[{}/{}] error en {}: {}", i, len(files), file.name, e)

    # Último grupo
    if group_rows:
        _print_group_summary(group_rows)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(config.OUTPUT_DIR / "experiment_results.parquet")
    logger.info("─" * 60)
    logger.info("Resultados guardados en {}", config.OUTPUT_DIR / "experiment_results.parquet")


if __name__ == "__main__":
    main()
