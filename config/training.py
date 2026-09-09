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
    # --- Bench (todas las familias de ALL_FAMILIES, hiperparámetros default + early
    # stopping, sin Optuna) ---
    bench_n_estimators: int
    # --- Optuna: única ronda del pipeline, solo sobre la familia ganadora, post-feature-selection ---
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
    bench_n_estimators=500,
    optuna_n_trials=100,
    optuna_timeout_s=180,
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
    bench_n_estimators=1_500,
    optuna_n_trials=200,
    optuna_timeout_s=5 * 60,
    optuna_n_estimators=1_500,
    cv_folds=3,
    final_n_estimators=5_000,
)

# eficiente: mismo flujo completo que "optimized" (baselines estadísticos, feature
# selection, SHAP, ronda larga de Optuna -- todas las fases ON, así el informe tiene
# tabla de features/hiperparámetros/SHAP para cada nivel/grain), pero con presupuesto
# de Optuna y de CV recortado a propósito ("deuda de tiempo de optimización" asumida
# conscientemente) para poder correr los 12 niveles x 2 grains (daily/weekly) en una
# sola pasada sin que cada combinación tome lo mismo que "optimized". Los números
# resultantes son reales (no un smoke-test) pero mejorables con más tiempo de ajuste
# -- dejar esa salvedad explícita en el informe.
EFFICIENT = TrainingProfile(
    name="efficient",
    run_baseline_stats=True,
    run_feature_selection=True,
    run_shap=True,
    run_optuna=True,
    bench_n_estimators=800,
    optuna_n_trials=100,
    optuna_timeout_s=120,
    optuna_n_estimators=800,
    cv_folds=1,
    final_n_estimators=1_500,
)

# optimizado: pipeline completo (todas las fases, presupuesto de Optuna
# original) -- la corrida "de calidad" para las métricas finales del TFM.
OPTIMIZED = TrainingProfile(
    name="optimized",
    run_baseline_stats=True,
    run_feature_selection=True,
    run_shap=True,
    run_optuna=True,
    bench_n_estimators=1_500,
    optuna_n_trials=500,
    optuna_timeout_s=10 * 60,
    optuna_n_estimators=1_500,
    cv_folds=3,
    final_n_estimators=5_000,
)

TRAINING_PROFILES = {p.name: p for p in (FAST, MODERATE, EFFICIENT, OPTIMIZED)}
