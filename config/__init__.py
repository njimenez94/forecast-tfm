"""Configuración del proyecto, separada por responsabilidad.

- `paths`    : rutas de datos, base de datos y artefactos.
- `features` : target, horizonte y configuración de features.
- `levels`   : registro de niveles de agregación (L1 → L12).
- `model`    : hiperparámetros y entrenamiento de LightGBM.
- `training` : perfiles de entrenamiento (fast/moderate/efficient/optimized) para scripts/train_dataset.py.

Todos los parámetros se reexportan aquí (explícito, no `import *`) para acceder
con `config.<NOMBRE>`; el archivo que define cada uno queda visible acá mismo.
"""
from config.paths import (
    ROOT, COMPETITION, RAW_DIR, RAW_ZIP, DB_PATH, CREATE_DATABASE_QUERY, PROCESSED_DIR,
    OUTPUT_DIR, ARTIFACTS_DIR, MODELS_DIR, LOGS_DIR, DATASETS,
    dataset_level_path, featured_level_path, featured_level_combos,
)
from config.features import (
    TARGET, HORIZON, SEASON_LENGTH, SN_WINDOWS, MA_WINDOWS, VALID_PERIODS, TEST_PERIODS,
    VALID_YEAR_BLOCKS, to_days, valid_days, valid_year_days, test_days, SPLIT_DATES,
    MLFORECAST_LAGS, MLFORECAST_LAG_TRANSFORMS, MLFORECAST_DATE_FEATURES, MLFORECAST_FREQ,
    EXCLUDE_AS_STATIC, PRICE_LAG_COL, EXOG_COLS,
)
from config.levels import Level, LEVELS, LEVELS_BY_ID, ACTIVE_LEVEL_IDS, ACTIVE_LEVELS
from config.model import SEED, TWEEDIE_LEVELS, TWEEDIE_VARIANCE_POWER, LGBM_PARAMS, lgbm_params
from config.training import TrainingProfile, FAST, MODERATE, EFFICIENT, OPTIMIZED, TRAINING_PROFILES
