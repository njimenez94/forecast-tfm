# forecast-tfm

Pipeline de forecasting de demanda (dataset **M5**) con LightGBM (vía mlforecast) y validación temporal.

Flujo: descargar datos de Kaggle → crear DB → construir datasets por nivel → generar features derivadas → entrenar/evaluar por nivel → generar test set final.

## Configuración

`make create-database` descarga el dataset de la competencia [m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) directo desde la API de Kaggle (no se versiona en git, ver reglas de la competencia). Necesitas tu propio token de Kaggle antes de correrlo:

```bash
export KAGGLE_API_TOKEN=tu_token   # generado en https://www.kaggle.com/settings/api
```

(alternativa: `uv run kaggle auth login` para autenticar vía OAuth en el navegador). También debes haber aceptado las reglas de la competencia en la página de Kaggle con tu cuenta.

## Uso típico

```bash
make create-database                          # 1. descarga backup/*.zip desde Kaggle -> data/m5.db
make process-data                              # 2. datasets base por nivel -> data/processed/
make build-datasets                            # 3. features derivadas -> data/datasets/
make train-dataset                             # 4. entrena y evalúa -> artifacts/{results,plots,models}/
make test-sets                                 # 5. predicciones finales + métricas -> artifacts/test_sets/, output/test_metrics.json
```

`make pipeline` encadena 2-4 (`process-data build-datasets train-dataset`); `create-database` y `test-sets` corren aparte. Sin `ARGS`, cada script usa los defaults de `config/` (niveles activos `config.ACTIVE_LEVEL_IDS`). Para filtrar: `ARGS="--levels 1,9,12"` (sin espacios entre comas), `ARGS="--profile fast"` (ver [Perfiles de entrenamiento](#perfiles-de-entrenamiento)) — combinables, p.ej. `ARGS="--levels 1 --profile fast"`.

## Cómo correrlo desde cero (para reproducirlo en otra PC)

Requisitos: [uv](https://docs.astral.sh/uv/) instalado y una cuenta de Kaggle que haya aceptado las reglas de la competencia [m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

```bash
git clone <repo> && cd forecast-tfm
uv sync                                        # instala Python 3.13 (ver .python-version) y dependencias

export KAGGLE_API_TOKEN=tu_token               # generado en https://www.kaggle.com/settings/api
# (alternativa: uv run kaggle auth login, autentica por navegador)

make create-database                           # descarga y arma data/m5.db (~2GB, tarda según tu conexión)
make process-data
make build-datasets
make train-dataset ARGS="--profile fast"       # ver nota abajo sobre el tiempo que tarda esto
make test-sets                                 # opcional: predicciones finales + métricas de test
```

**Sobre el tiempo de `train-dataset`:** por defecto entrena **todos los niveles activos** (`config.ACTIVE_LEVEL_IDS`, hoy 1,4,6,9,10,12), cada uno con bench de las 5 familias (hiperparámetros default, sin Optuna) + Optuna sobre la ganadora + SHAP, así que sin acotar algo puede tardar bastante en los niveles grandes (10-12, un modelo por combinación).

Para una primera corrida (o para solo verificar que el pipeline anda de punta a punta):

- `ARGS="--profile fast"`: perfil de smoke-test, salta baselines estadísticos/feature-selection/SHAP/Optuna final (ver [Perfiles de entrenamiento](#perfiles-de-entrenamiento)).
- Sumar `--levels 1` para acotar todavía más a una sola corrida.
- Corrida larga (horas): dejarla corriendo en `tmux`/`nohup` en vez de la terminal interactiva, así sobrevive si se cierra la sesión — el progreso queda igual en `logs/train_dataset/` (ver tabla de Pipeline más abajo).

Los artifacts livianos por nivel quedan en `artifacts/models/*.pkl` (sí están versionados en git, a diferencia del resto de `artifacts/` y de `data/`, ver [Estructura](#estructura)).

## Perfiles de entrenamiento

`train-dataset` agrupa las fases on/off y el presupuesto de Optuna en cuatro perfiles (`config/training.py`), elegibles con `--profile` (default: la constante `PROFILE` en [scripts/train_dataset/config.py](scripts/train_dataset/config.py), hoy `"efficient"`). El bench de familias (`benchmarking.run_bench_ml`) nunca usa Optuna en ningún perfil — hiperparámetros default + early stopping, para dejarle el presupuesto de Optuna a la ronda final sobre la ganadora:

| Perfil | Fases | Uso |
|---|---|---|
| `fast` | Sin baselines estadísticos, feature selection, SHAP ni Optuna. Un solo split (cv_folds=1, sin rolling CV). | Smoke-test: verificar que un nivel corre entero. No pensado para producción. |
| `moderate` | Sin baselines estadísticos/feature selection/SHAP (se corren después solo sobre el ganador). Optuna final recortado; fit final con la calidad completa (rolling CV, 5000 árboles). | Primera versión entrenable en producción sin esperar el pipeline completo. |
| `efficient` | Mismo flujo completo que `optimized` (todas las fases), pero con presupuesto de Optuna y de CV recortado a propósito para correr los 12 niveles x 2 grains en una sola pasada. Números reales (no smoke-test) pero mejorables con más tiempo de ajuste. | Corrida completa acotada en tiempo — default actual. |
| `optimized` | Pipeline completo: todas las fases, presupuesto de Optuna original. | Corrida de calidad para las métricas finales del TFM. |

Para agregar un parámetro nuevo al esquema de `Config` (no para bajar la calidad de una corrida puntual, para eso está `--profile`) se edita `Config` directamente en `scripts/train_dataset/config.py`.

## Pipeline (script → make → salida)

| # | Script | `make` | Qué hace | Salida |
|---|--------|--------|----------|--------|
| 1 | [scripts/create_database.py](scripts/create_database.py) | `create-database` | Descarga `backup/m5-forecasting-accuracy.zip` desde Kaggle si no existe, lo descomprime en `data/raw/` y crea la base DuckDB. | `data/m5.db` |
| 2 | [scripts/process_data.py](scripts/process_data.py) | `process-data` | Por nivel de agregación (L1→L12) y granularidad (daily/weekly), extrae ventas (`sales`, precio, calendario, eventos). `ARGS="--levels 1,9,12"` (default: `config.ACTIVE_LEVEL_IDS`), `ARGS="--counts"` (solo contar, sin materializar). | `data/processed/{daily,weekly}/level_*.parquet` |
| 3 | [scripts/build_datasets.py](scripts/build_datasets.py) | `build-datasets` | Aplica `src.features.pipeline` (calendario, precio, eventos, encoding, intermitencia, lags/rolling/momentum) sobre cada parquet de `data/processed/`. `ARGS="--levels 1,9,12"` (default: `config.ACTIVE_LEVEL_IDS`). | `data/datasets/{daily,weekly}/dataset_level_*.parquet` |
| 4 | [scripts/train_dataset/](scripts/train_dataset/) | `train-dataset` | Entrena LightGBM/XGBoost/CatBoost/HistGB/Ridge por nivel/grain sobre `sales` (comparación de baselines, selección de features, SHAP, tuning Optuna sobre la ganadora). `ARGS="--levels 1,9,12"` (default: `config.ACTIVE_LEVEL_IDS`), `ARGS="--profile fast\|moderate\|efficient\|optimized"` (default: ver [Perfiles de entrenamiento](#perfiles-de-entrenamiento)). Niveles 10-12 entrenan un modelo por combinación `split_by` (dept/store). Logs en `logs/train_dataset/`. | `artifacts/results/*_{model_comparison,series_metrics,feature_selection}.csv`, `artifacts/plots/*_shap_*.png`, `artifacts/optuna_study.db`, `artifacts/models/*_artifact.pkl` |
| 5 | [scripts/build_test_sets.py](scripts/build_test_sets.py) | `test-sets` | Corre el modelo final de cada nivel sobre su test set y agrega las métricas de error. | `artifacts/test_sets/{level}.parquet`, `output/test_metrics.json` |

Métricas: WAPE, Bias, WRMSSE (y por serie también MASE/SPEC) — ver [src/evaluation/](src/evaluation/).

## Estructura

```
backup/     Zip de Kaggle descargado por create-database (no versionado, ver Configuración)
data/       raw/ (CSV de M5), processed/ (por nivel, sin features), datasets/ (con features, listo para entrenar), m5.db — solo se versiona la estructura
config/     Parámetros del proyecto separados por responsabilidad (paths, features, levels, model, training)
queries/    SQL para crear la base DuckDB y los datasets base
src/        Lógica reutilizable (datos, features, modelado, evaluación)
scripts/    Puntos de entrada ejecutables (orquestan src/)
notebooks/  Exploración (eda) y prototipado del modelo (model)
artifacts/  results/, plots/, models/, test_sets/ y optuna_study.db — generados por train-dataset y test-sets
output/     test_metrics.json — métricas del modelo final sobre test (test-sets)
logs/       Un .log por corrida (timestamp), en una subcarpeta por script (process_data/, build_datasets/, train_dataset/, etc.)
```
