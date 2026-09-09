"""Gate de no-regresión: compara las últimas dos versiones registradas en
artifacts/models/registry.json para un nivel/target y falla (exit 1) si
wrmsse_test o wape_test empeoraron más de `--tolerance` respecto a la versión
anterior.

No corre en CI: necesita un registry.json real, generado entrenando sobre datos de
Kaggle (make train-dataset). Es un gate manual antes de mergear un cambio en
src/features/ o src/modeling/ -- ver README, sección de testing.

Uso:
    python -m scripts.check_regression --level 1
    python -m scripts.check_regression --level 1 --target sales --tolerance 0.10
"""
import argparse
import json
import sys

import config


def _registry_key(registry: dict, level_id: int, target: str) -> str | None:
    prefix = f"level_{level_id:02d}_"
    for key in registry:
        if key.startswith(prefix) and key.endswith(f"_{target}"):
            return key
    return None


def check_regression(entry: dict, tolerance: float,
                      metrics: tuple[str, ...] = ("wrmsse_test", "wape_test")) -> list[str]:
    """`entry` es `registry[key]` (con "versions", ver export.py::_update_registry).
    Compara las últimas dos versiones por timestamp (las claves de "versions" son
    UTC "%Y%m%dT%H%M%SZ", ordenables lexicográficamente igual que cronológicamente).
    Devuelve una lista de mensajes de regresión -- vacía si no hay ninguna, o si hay
    menos de 2 versiones para comparar (nada contra qué regresionar todavía)."""
    versions = sorted(entry.get("versions", {}).items())
    if len(versions) < 2:
        return []
    (_, previous), (latest_version, latest) = versions[-2:]

    violations = []
    for metric in metrics:
        old_value, new_value = previous.get(metric), latest.get(metric)
        if old_value is None or new_value is None:
            continue
        if new_value > old_value * (1 + tolerance):
            pct = (new_value / old_value - 1) * 100
            violations.append(
                f"{metric} empeoró {pct:.1f}% ({old_value:.4f} -> {new_value:.4f}, versión {latest_version})",
            )
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, required=True, help="level_id (ver config/levels.py)")
    ap.add_argument("--target", default="sales")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="Fracción de degradación tolerada antes de fallar (default 0.05 = 5%%).")
    args = ap.parse_args()

    registry_path = config.MODELS_DIR / "registry.json"
    if not registry_path.exists():
        print(f"No hay registry.json en {registry_path} -- nada que comparar todavía.")
        return 0

    registry = json.loads(registry_path.read_text())
    key = _registry_key(registry, args.level, args.target)
    if key is None:
        print(f"Nivel {args.level} (target={args.target}) sin versiones registradas todavía.")
        return 0

    violations = check_regression(registry[key], args.tolerance)
    if violations:
        print(f"REGRESIÓN detectada en {key} (tolerancia {args.tolerance:.0%}):")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(f"Sin regresión en {key} (tolerancia {args.tolerance:.0%}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
