"""Config de una corrida de entrenamiento. Todo lo que controla el pipeline vive
acá, agrupado por fase -- editar y correr (ver __main__.py)."""
from dataclasses import dataclass, field

import config


@dataclass
class Config:
    # --- selección de dataset ---
    level_id: int = 1
    grain: str = "daily"  # "daily" o "weekly" -- debe estar en level.grains (config/levels.py)

    # --- fases on/off ---
    # Defaults tomados de config.OPTIMIZED (única fuente de verdad, ver
    # config/training.py) -- así este dataclass no repite a mano los valores del
    # perfil. --profile (__main__.py) los pisa con los de config.TRAINING_PROFILES
    # antes de cada corrida igual; para bajar la calidad de una corrida puntual usar
    # --profile fast/moderate, no editar los defaults acá.
    run_baseline_naive: bool = True
    run_baseline_stats: bool = config.OPTIMIZED.run_baseline_stats  # SARIMA/ETS/Theta/TBATS/Prophet: serie x serie, lento
    # LightGBM/XGBoost/CatBoost/HistGB/Ridge, cada uno con hiperparámetros default +
    # early stopping (sin Optuna, ver benchmarking.run_bench_ml). El ganador (mejor
    # familia tree-based) sigue a feature selection / SHAP / Optuna final.
    run_bench_ml: bool = True
    run_feature_selection: bool = config.OPTIMIZED.run_feature_selection  # permutation importance + backward elimination
    run_shap: bool = config.OPTIMIZED.run_shap
    run_optuna: bool = config.OPTIMIZED.run_optuna  # única ronda de Optuna, post-feature-selection, solo sobre el ganador
    save_artifact: bool = True

    # --- comparación de modelos base ---
    random_state: int = 42

    # --- selección de features ---
    permutation_sample_size: int = 15_000
    permutation_n_repeats: int = 5
    # Relativo al score actual (ver backward_feature_selection), no absoluto -- así
    # es comparable entre niveles con escalas de objective_metric muy distintas.
    # OJO: >0 hace que el criterio de aceptación "arrastre" (best_score se
    # actualiza a cada aceptación, incluso si es levemente peor que el anterior,
    # así que el degrade tolerado se acumula paso a paso sin techo) -- esto ya
    # pasaba con el algoritmo original uno-a-uno, no es nuevo del batching, pero
    # con tolerance>0 el ORDEN de las pruebas importa mucho más: batch vs.
    # uno-a-uno pueden terminar seleccionando conjuntos de features bien
    # distintos (probado: 6 vs. 21 features en un caso sintético).
    # Subido de 0.0 a 0.005 y luego a 0.02 (2026-09-16): con 0.0, cualquier ruido
    # numérico entre reentrenamientos rechaza el bloque de 8 y cae al fallback
    # uno-a-uno (ver "Batching" abajo) -- en la práctica casi ningún bloque
    # prosperaba, así que el batching no aportaba nada y "selección de features"
    # terminaba siendo ~85% del tiempo total del pipeline (ver
    # logs/train_dataset/20260915_223756.log). 0.005 mejoró poco: un bloque
    # rechazado paga el intento del bloque + cada feature individual (el mismo
    # costo que sin batching, más uno de yapa), así que si la mayoría de los
    # bloques se siguen rechazando, batching no ahorra nada. 0.02 (2% relativo)
    # hace que los bloques de features poco importantes pasen la mayoría de las
    # veces, así se paga 1 reentrenamiento por bloque en vez de N -- más agresivo
    # podando, pero es justo lo que se pidió (descartar lo obvio rápido, no
    # selección fina).
    backward_tolerance: float = 0.02
    # Cuántas features candidatas se prueba remover juntas en cada reentrenamiento
    # de backward_feature_selection (ver docstring ahí, sección "Batching"). 1 =
    # una por una (algoritmo original, más lento con muchas features); >1 = por
    # lotes, con fallback automático a uno-a-uno si el lote se rechaza. Más rápido,
    # y en la práctica tiende a podar mejor grupos de features redundantes entre
    # sí -- pero la selección final puede diferir de la de max_batch_size=1 (no
    # es un cambio de criterio de aceptación, es que un bloque completo es una
    # prueba más exigente que sus features por separado). Para reproducir bit a
    # bit una corrida sin batching, usar 1.
    # Subido de 8 a 16 (2026-09-16): junto con backward_tolerance=0.005, bloques
    # más grandes podan más features por reentrenamiento, menos iteraciones totales.
    feature_selection_batch_size: int = 16

    # --- SHAP ---
    shap_sample_size: int = 300_000

    # --- Bench (TODAS las familias de ALL_FAMILIES, hiperparámetros default + early
    # stopping, sin Optuna -- ver benchmarking.run_bench_ml) ---
    bench_n_estimators: int = config.OPTIMIZED.bench_n_estimators

    # --- Optuna: única ronda del pipeline, solo sobre la familia ganadora, post-feature-selection ---
    optuna_n_trials: int = config.OPTIMIZED.optuna_n_trials
    optuna_timeout_s: int = config.OPTIMIZED.optuna_timeout_s
    optuna_n_estimators: int = config.OPTIMIZED.optuna_n_estimators

    # --- modelo final ---
    # Folds rolling-origin sobre train+valid para elegir n_estimators antes del
    # refit final sobre todos los datos (ver rolling_cv_folds). 1 = sin CV, un solo
    # split como antes.
    cv_folds: int = config.OPTIMIZED.cv_folds
    final_n_estimators: int = config.OPTIMIZED.final_n_estimators
    objective_by_level: dict = field(default_factory=lambda: {12: "tweedie"})
    default_objective: str = "regression_l2"

    @property
    def objective(self) -> str:
        return self.objective_by_level.get(self.level_id, self.default_objective)

    # Fijo si run_optuna=False (fit_final_model cae al fallback de bench_params en
    # ese caso, ver refinement.py); si run_optuna=True y objective=="tweedie", Optuna
    # tunea este valor (1.1-1.9) para las familias que lo soportan (lightgbm/xgboost)
    # en vez de usar el fijo.
    tweedie_variance_power: float = 1.5


CFG = Config()

# Perfil que se usa cuando se corre sin --profile (ver config/training.py). Editar
# acá, junto con CFG, para cambiar el default sin tener que pasar el flag.
PROFILE = "efficient"  # fast moderate efficient optimized
