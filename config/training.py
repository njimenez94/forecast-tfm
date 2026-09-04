"""Perfiles de entrenamiento para scripts/train_dataset.py.

Agrupan las fases on/off y el presupuesto de Optuna que hacen a un run "rápido",
"moderado" u "optimizado", para no tener que editar `Config` a mano cada vez que
se busca un balance distinto entre velocidad y calidad. Elegir con `--profile`
(ver scripts/train_dataset.py:main()).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    # --- fases on/off (ver Config en scripts/train_dataset.py) ---
    run_baseline_stats: bool
    run_feature_selection: bool
    run_shap: bool
    run_optuna: bool
    # --- Optuna: bench (todas las familias de ALL_FAMILIES, corto) ---
    optuna_bench_n_trials: int
    optuna_bench_timeout_s: int
    # --- Optuna: final (largo, solo sobre la familia ganadora, post-feature-selection) ---
    optuna_n_trials: int
    optuna_timeout_s: int
    optuna_n_estimators: int
    # --- modelo final ---
    cv_folds: int
    final_n_estimators: int


# rápido: smoke-test del pipeline / iterar rápido en dev. Sin baselines
# estadísticos, sin feature selection/SHAP, sin ronda larga de Optuna, bench
# mínimo y un solo split (cv_folds=1, sin rolling CV). No pensado para modelos
# que vayan a producción -- solo para validar que un nivel/target corre entero.
FAST = TrainingProfile(
    name="fast",
    run_baseline_stats=False,
    run_feature_selection=False,
    run_shap=False,
    run_optuna=False,
    optuna_bench_n_trials=20,
    optuna_bench_timeout_s=60,
    optuna_n_trials=0,
    optuna_timeout_s=0,
    optuna_n_estimators=500,
    cv_folds=1,
    final_n_estimators=1_000,
)

# moderado: primera versión "en producción" sin las fases caras de
# interpretabilidad/comparación (baselines estadísticos, feature selection,
# SHAP -- esas se corren después, solo sobre el modelo ganador, para el TFM).
# Bench y Optuna final recortados, pero el fit final mantiene la calidad
# (rolling CV de 3 folds, 5000 árboles).
MODERATE = TrainingProfile(
    name="moderate",
    run_baseline_stats=False,
    run_feature_selection=False,
    run_shap=False,
    run_optuna=True,
    optuna_bench_n_trials=40,
    optuna_bench_timeout_s=90,
    optuna_n_trials=200,
    optuna_timeout_s=5 * 60,
    optuna_n_estimators=1_500,
    cv_folds=3,
    final_n_estimators=5_000,
)

# optimizado: pipeline completo (todas las fases, presupuesto de Optuna
# original) -- la corrida "de calidad" para las métricas finales del TFM.
OPTIMIZED = TrainingProfile(
    name="optimized",
    run_baseline_stats=True,
    run_feature_selection=True,
    run_shap=True,
    run_optuna=True,
    optuna_bench_n_trials=200,
    optuna_bench_timeout_s=5 * 60,
    optuna_n_trials=500,
    optuna_timeout_s=10 * 60,
    optuna_n_estimators=1_500,
    cv_folds=3,
    final_n_estimators=5_000,
)

TRAINING_PROFILES = {p.name: p for p in (FAST, MODERATE, OPTIMIZED)}
