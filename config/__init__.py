"""Configuración del proyecto, separada por responsabilidad.

- `paths`    : rutas de datos, base de datos y artefactos.
- `features` : target, horizonte y configuración de features.
- `levels`   : registro de niveles de agregación (L1 → L12).
- `model`    : hiperparámetros y entrenamiento de LightGBM.
- `training` : perfiles de entrenamiento (fast/moderate/optimized) para scripts/train_dataset.py.

Todos los parámetros se reexportan aquí para acceder con `config.<NOMBRE>`.
"""
from config.paths import *  # noqa: F401,F403
from config.features import *  # noqa: F401,F403
from config.levels import *  # noqa: F401,F403
from config.model import *  # noqa: F401,F403
from config.training import *  # noqa: F401,F403
