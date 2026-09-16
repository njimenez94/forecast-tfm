"""Reconstruye artifacts/models/registry.json (+ su carpeta versions/) a partir de
los artifacts planos `{level_str}_{target}_artifact.pkl`, sin reentrenar.

Por qué hace falta: registry.json y artifacts/models/versions/ están en .gitignore
(rolling, "se regenera reentrenando" -- ver comentario en .gitignore), así que un
`git pull` de artifacts entrenados en otra máquina (p.ej. la VM de entrenamiento)
trae los .pkl planos pero NO trae su versión ni su entrada de registry: esas viven
solo en el filesystem local de la máquina que corrió export_artifact. Si esta
máquina había entrenado antes localmente algunos niveles, su registry.json queda
con entradas *viejas* que ya no corresponden al .pkl plano actual (mismo
winner_family pero métricas de test distintas -- confirmado a mano para los 3 keys
que ya tenían entrada antes de este script).

Cada .pkl plano trae toda la info que pide `_update_registry` (winner_family,
*_test) -- ver src/training/export.py -- así que alcanza con leerlo. Lo único que
export_artifact no persiste en el artifact es el timestamp de entrenamiento (el
"version" es la hora de la corrida, no un campo del dict); como proxy se usa la
fecha del commit de git que trajo ese .pkl (mejor información disponible sin
reentrenar), con el mtime del archivo como fallback si no está trackeado.

Idempotente: si la key ya tiene registrada una versión con ese mismo timestamp,
la saltea. No pisa versiones previas de la misma key (se agregan al historial),
solo actualiza "latest".

Uso:
    python -m scripts.backfill_registry
    python -m scripts.backfill_registry --level 9
    python -m scripts.backfill_registry --dry-run
"""
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
from loguru import logger

import config

REGISTRY_PATH = config.MODELS_DIR / "registry.json"
VERSIONS_DIR = config.MODELS_DIR / "versions"


def _git_commit_time(path: Path) -> datetime | None:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", str(path)],
        cwd=config.ROOT, capture_output=True, text=True, check=False,
    )
    stamp = result.stdout.strip()
    if not stamp:
        return None
    return datetime.fromisoformat(stamp).astimezone(timezone.utc)


def _version_for(path: Path) -> str:
    when = _git_commit_time(path) or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return when.strftime("%Y%m%dT%H%M%SZ")


def backfill(path: Path, registry: dict, dry_run: bool) -> bool:
    """Devuelve True si agregó una versión nueva para `path` en `registry` (in-place)."""
    key = path.stem.removesuffix("_artifact")
    version = _version_for(path)

    entry = registry.setdefault(key, {"latest": None, "versions": {}})
    if version in entry["versions"]:
        logger.info("[skip] {} ya tiene la versión {}", key, version)
        return False

    artifact = joblib.load(path)
    versioned_path = VERSIONS_DIR / f"{key}_artifact_{version}.pkl"

    logger.info(
        "[{}] {} -> versión {} (winner={}, wape_test={:.2%}, wrmsse_test={:.4f})",
        "dry-run" if dry_run else "ok", key, version,
        artifact.get("winner_family"), artifact.get("wape_test", float("nan")),
        artifact.get("wrmsse_test", float("nan")),
    )
    if dry_run:
        return True

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, versioned_path)

    entry["versions"][version] = {
        "path": str(versioned_path.relative_to(config.ROOT)),
        "trained_at": version,
        "winner_family": artifact.get("winner_family"),
        **{k: v for k, v in artifact.items() if k.endswith("_test")},
    }
    entry["latest"] = version
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, default=None, help="Solo backfillear este level_id (default: todos)")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar qué haría sin escribir nada")
    args = parser.parse_args()

    pattern = f"level_{args.level:02d}_*_artifact.pkl" if args.level else "*_artifact.pkl"
    paths = sorted(config.MODELS_DIR.glob(pattern))
    logger.info("{} artifacts encontrados ({})", len(paths), pattern)

    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}
    updated = sum(backfill(path, registry, args.dry_run) for path in paths)

    if updated and not args.dry_run:
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
        logger.success("registry.json actualizado ({} keys)", len(registry))
    logger.info("{}/{} artifacts con versión nueva", updated, len(paths))


if __name__ == "__main__":
    main()
