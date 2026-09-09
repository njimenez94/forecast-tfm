import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
from loguru import logger

import config
from src.modeling import MODEL_FAMILIES, compute_permutation_importance
from scripts.train_dataset.config import Config


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


def export_artifact(state: SimpleNamespace, cfg: Config) -> None:
    import shap

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

    # TreeExplainer se construye acá (sobre el estimador nativo, no el FittedModel
    # wrapper -- ver families.py) y se guarda ya armado en el artifact: todos los
    # consumidores (notebooks, api/) lo cargan listo en vez de reconstruirlo cada vez,
    # que además requeriría acordarse de pasar `model.estimator` en lugar de `model`
    # (todas las familias ganadoras son tree-based, ver pick_winner).
    explainer = shap.TreeExplainer(state.final_model.estimator)

    # Solo modelo + metadata (features, métricas): liviano para respaldar/mover entre
    # máquinas sin arrastrar datos. df/X_*/y_*/train/valid/test NO se guardan --
    # src.data.dataset.reconstruct_test_data() las recrea desde el parquet en
    # data/datasets/ repitiendo load_data()+split_data() (ver notebooks/03_predictions.ipynb).
    artifact = {
        "level": state.level_str,
        "level_id": state.level.id,
        "winner_family": state.winner_family,
        "model": state.final_model,
        "explainer": explainer,
        "model_params": state.model_params,
        "features": state.features,
        "categorical_features": state.categorical_features,
        # Dominio de categorías vistas en training por columna -- api/main.py lo necesita
        # para reconstruir el dtype category de un payload de 1 fila: sin esto, una
        # columna con valor faltante en esa fila (p.ej. event_name_2, casi siempre nula)
        # queda con categories=[] y XGBoost rechaza la predicción ("must have at least
        # one category"); cualquier placeholder no visto en training también falla
        # ("category not in the training set"). El valor de la fila sigue siendo NaN
        # (missing) -- solo hace falta que la lista de categorías declarada sea real.
        "categorical_categories": {
            col: state.X_train[col].cat.categories.tolist() for col in state.categorical_features
        },
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
        # Config completa de la corrida (perfil de fases, presupuesto de Optuna,
        # params de feature selection, random_state, ...) -- antes solo quedaban sus
        # *efectos* (features, model_params, métricas). Guardarla entera permite
        # diffear dos versiones de registry.json para explicar un cambio de métrica
        # sin adivinar qué perfil/flags corrieron cada una (ver plan de
        # reproducibilidad, Fase 5).
        "config": asdict(cfg),
    }

    out_dir = config.MODELS_DIR
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
