"""Resuelve y cachea en memoria los artifacts entrenados (ver
scripts/train_dataset.py::export_artifact) para servirlos desde la API.

Dos fuentes, en este orden de preferencia:
- artifacts/models/registry.json: versiones explícitas escritas por export_artifact
  desde que se agregó el versionado (clave "latest" + historial en "versions").
- Fallback al archivo plano `{level_str}_{target}_artifact.pkl` que export_artifact
  siempre sobreescribe, para niveles entrenados antes de que existiera el registry.
"""
import json
from pathlib import Path

import joblib

import config

REGISTRY_PATH = config.MODELS_DIR / "registry.json"

_cache: dict[tuple[int, str, str | None], dict] = {}


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text())


def _artifacts_subdir(target: str) -> Path:
    return config.MODELS_DIR if target == "sales" else config.MODELS_DIR / target


def _registry_key(level_id: int, target: str) -> str | None:
    """La clave del registry es `{level_str}_{target}`; level_str no es derivable de
    level_id solo (incluye grain/nombre), así que se resuelve buscando la entrada
    cuya versión más reciente apunte a un path con el prefijo `level_{level_id:02d}_`."""
    prefix = f"level_{level_id:02d}_"
    for key in load_registry():
        if key.startswith(prefix) and key.endswith(f"_{target}"):
            return key
    return None


def resolve_artifact_path(level_id: int, target: str = "sales", version: str | None = None) -> Path:
    if level_id not in config.LEVELS_BY_ID:
        raise KeyError(f"Nivel desconocido: {level_id}")

    registry = load_registry()
    key = _registry_key(level_id, target)
    if key is not None:
        entry = registry[key]
        resolved_version = version or entry["latest"]
        if resolved_version not in entry["versions"]:
            raise KeyError(f"Versión desconocida para {key}: {resolved_version}")
        return config.ROOT / entry["versions"][resolved_version]["path"]

    if version is not None:
        raise KeyError(f"Nivel {level_id} no tiene versiones registradas en registry.json")

    # Fallback: nivel entrenado antes del versionado, solo existe el archivo plano.
    matches = sorted(_artifacts_subdir(target).glob(f"level_{level_id:02d}_*_{target}_artifact.pkl"))
    if not matches:
        raise FileNotFoundError(f"No hay artifact para nivel {level_id}, target {target}")
    return matches[0]


def resolve_version(level_id: int, target: str = "sales", version: str | None = None) -> str:
    key = _registry_key(level_id, target)
    if key is not None:
        registry = load_registry()
        return version or registry[key]["latest"]
    return version or "unversioned"


def get_model(level_id: int, target: str = "sales", version: str | None = None) -> dict:
    cache_key = (level_id, target, version)
    if cache_key not in _cache:
        path = resolve_artifact_path(level_id, target, version)
        _cache[cache_key] = joblib.load(path)
    return _cache[cache_key]
