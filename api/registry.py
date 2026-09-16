"""Resuelve y cachea en memoria los artifacts entrenados (ver
scripts/train_dataset.py::export_artifact) para servirlos desde la API.

Dos fuentes, en este orden de preferencia (por nivel+grain+target, no por nivel+target
solo -- ver GRAIN más abajo):
- artifacts/models/registry.json: versiones explícitas escritas por export_artifact
  desde que se agregó el versionado (clave "latest" + historial en "versions").
- Fallback al archivo plano `{level_str}_{target}_artifact.pkl` que export_artifact
  siempre sobreescribe, para niveles/grains entrenados antes de que existiera el
  registry (o simplemente no vueltos a entrenar desde entonces).

GRAIN: cada level_id soporta dos granularidades (daily/weekly, ver
config.Level.grains) y `train-dataset` sin `--grain` entrena las dos por defecto --
hoy conviven ambos archivos planos para casi todos los niveles activos
(`level_XX_daily_..._artifact.pkl` y `level_XX_weekly_..._artifact.pkl`), aunque
registry.json todavía solo trackee uno de los dos para algunos. Por eso `grain` es
un parámetro explícito en toda esta resolución: sin él, si existe más de un grain
para ese level_id+target, se levanta ValueError en vez de elegir uno en silencio
(antes se resolvía por orden de inserción/alfabético, sirviendo el modelo
equivocado sin avisar)."""
import json
from pathlib import Path

import joblib

import config

REGISTRY_PATH = config.MODELS_DIR / "registry.json"
GRAINS = ("daily", "weekly")

_cache: dict[tuple[int, str, str | None, str | None], dict] = {}


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text())


def _artifacts_subdir(target: str) -> Path:
    return config.MODELS_DIR if target == "sales" else config.MODELS_DIR / target


def _validate_grain(grain: str | None) -> None:
    if grain is not None and grain not in GRAINS:
        raise ValueError(f"grain inválido: {grain!r} (debe ser uno de {GRAINS})")


def _grain_of_key(level_id: int, key: str) -> str:
    # key = f"level_{level_id:02d}_{grain}_{name}_{target}" (ver export.py) -- el
    # primer segmento después del id es siempre el grain, sin importar cuántos "_"
    # traiga el name o el target.
    return key[len(f"level_{level_id:02d}_"):].split("_", 1)[0]


def _registry_keys(level_id: int, target: str, grain: str | None = None) -> list[str]:
    prefix = f"level_{level_id:02d}_"
    suffix = f"_{target}"
    keys = []
    for key in load_registry():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        if grain is not None and _grain_of_key(level_id, key) != grain:
            continue
        keys.append(key)
    return keys


def _registry_key(level_id: int, target: str, grain: str | None = None) -> str | None:
    """La clave del registry es `{level_str}_{target}`, con level_str =
    `level_{level_id:02d}_{grain}_{name}` (ver scripts/train_dataset/export.py). Como
    level_str no es derivable de level_id solo (incluye grain y name), se resuelve
    buscando la entrada cuya versión más reciente apunte a un path con el prefijo
    `level_{level_id:02d}_`, filtrando por `grain` si se especifica.

    Si `grain` es None y hay más de una entrada para ese level_id+target (un grain
    cada una), se levanta ValueError en vez de devolver una al azar -- el caller
    (api/main.py) debe pedir el grain explícito."""
    keys = _registry_keys(level_id, target, grain)
    if len(keys) > 1:
        grains_found = sorted({_grain_of_key(level_id, k) for k in keys})
        raise ValueError(
            f"Nivel {level_id} target {target!r} tiene {len(keys)} versiones registradas "
            f"({grains_found}) -- especificá `grain` ({GRAINS}) para desambiguar."
        )
    return keys[0] if keys else None


def resolve_artifact_path(level_id: int, target: str = "sales", version: str | None = None, grain: str | None = None) -> Path:
    if level_id not in config.LEVELS_BY_ID:
        raise KeyError(f"Nivel desconocido: {level_id}")
    _validate_grain(grain)

    registry = load_registry()
    key = _registry_key(level_id, target, grain)
    if key is not None:
        entry = registry[key]
        resolved_version = version or entry["latest"]
        if resolved_version not in entry["versions"]:
            raise KeyError(f"Versión desconocida para {key}: {resolved_version}")
        return config.ROOT / entry["versions"][resolved_version]["path"]

    if version is not None:
        raise KeyError(f"Nivel {level_id} (grain={grain}) no tiene versiones registradas en registry.json")

    # Fallback: nivel/grain entrenado antes del versionado (o no vuelto a entrenar
    # desde que se agregó el registry), solo existe el archivo plano.
    grain_glob = grain or "*"
    matches = sorted(_artifacts_subdir(target).glob(f"level_{level_id:02d}_{grain_glob}_*_{target}_artifact.pkl"))
    if grain is None:
        grains_found = sorted({_grain_of_key(level_id, m.stem) for m in matches})
        if len(grains_found) > 1:
            raise ValueError(
                f"Nivel {level_id} target {target!r} tiene artifacts sin registrar en "
                f"{grains_found} -- especificá `grain` ({GRAINS}) para desambiguar."
            )
    if not matches:
        raise FileNotFoundError(f"No hay artifact para nivel {level_id} (grain={grain}), target {target}")
    return matches[0]


def resolve_version(level_id: int, target: str = "sales", version: str | None = None, grain: str | None = None) -> str:
    key = _registry_key(level_id, target, grain)
    if key is not None:
        registry = load_registry()
        return version or registry[key]["latest"]
    return version or "unversioned"


def artifact_grain(artifact: dict) -> str:
    """Grain ("daily"/"weekly") del artifact ya cargado -- se lee de
    `level_label` (`level_{id:02d}_{grain}`, ver export.py) en vez de pedírselo de
    nuevo al caller, para que la respuesta de /predict siempre refleje el grain
    realmente servido aunque el caller no lo haya especificado."""
    return artifact["level_label"].rsplit("_", 1)[-1]


def get_model(level_id: int, target: str = "sales", version: str | None = None, grain: str | None = None) -> dict:
    cache_key = (level_id, target, version, grain)
    if cache_key not in _cache:
        path = resolve_artifact_path(level_id, target, version, grain)
        _cache[cache_key] = joblib.load(path)
    return _cache[cache_key]
