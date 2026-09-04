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
make train-dataset ARGS="--levels 1,9,12"      # 4. entrena y evalúa -> artifacts/{results,plots,models}/
make test-sets                                 # 5. predicciones finales + métricas -> artifacts/test_sets/, output/test_metrics.json
```

`make pipeline` encadena 2-4 (`process-data build-datasets train-dataset`); `create-database` y `test-sets` corren aparte. Los targets que aceptan filtros usan `ARGS="--levels 1,9,12"` (sin espacios entre comas).

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
make train-dataset ARGS="--levels 1,9,12"      # ver nota abajo sobre el tiempo que tarda esto
make test-sets                                 # opcional: predicciones finales + métricas de test
```

**Sobre el tiempo de `train-dataset`:** con la configuración por defecto (bench de todas las familias + Optuna corto y largo + SHAP por nivel) puede tardar horas o días según cuántos niveles se corran. Para solo verificar que el pipeline anda de punta a punta sin esperar tanto:

- Correr un solo nivel liviano: `ARGS="--levels 1"`.
- O reducir las fases/tiempos de tuning editando `CFG` en [scripts/train_dataset.py](scripts/train_dataset.py) (por ejemplo bajar `optuna_bench_timeout_s` / `optuna_timeout_s`, o poner `run_baseline_stats=False` para saltear los modelos estadísticos clásicos, que son los más lentos).

Los artifacts livianos por nivel quedan en `artifacts/models/*.pkl` (sí están versionados en git, a diferencia del resto de `artifacts/` y de `data/`, ver [Estructura](#estructura)).

## Pipeline (script → make → salida)

| # | Script | `make` | Qué hace | Salida |
|---|--------|--------|----------|--------|
| 1 | [scripts/create_database.py](scripts/create_database.py) | `create-database` | Descarga `backup/m5-forecasting-accuracy.zip` desde Kaggle si no existe, lo descomprime en `data/raw/` y crea la base DuckDB. | `data/m5.db` |
| 2 | [scripts/process_data.py](scripts/process_data.py) | `process-data` | Por nivel de agregación (L1→L12) y granularidad (daily/weekly), extrae ventas + targets acumulados (`cumN`). `ARGS="--levels 1,9,12"`, `ARGS="--counts"` (solo contar, sin materializar). | `data/processed/{daily,weekly}/level_*.parquet` |
| 3 | [scripts/build_datasets.py](scripts/build_datasets.py) | `build-datasets` | Aplica `src.features` (lags, rolling, momentum) sobre cada parquet de `data/processed/`. `ARGS="--levels 1,9,12"`. | `data/datasets/{daily,weekly}/dataset_level_*.parquet` |
| 4 | [scripts/train_dataset.py](scripts/train_dataset.py) | `train-dataset` | Entrena LightGBM por nivel/grain/target (comparación de baselines, selección de features, SHAP, tuning Optuna). `ARGS="--levels 1,9,12"`, `ARGS="--target cum28"` (default: `sales`). Niveles 10-12 entrenan un modelo por combinación `split_by` (dept/store). | `artifacts/results[/{target}]/*_{model_comparison,series_metrics,feature_selection}.csv`, `artifacts/plots[/{target}]/*_shap_*.png`, `artifacts/optuna_study.db`, `artifacts/models[/{target}]/*_artifact.pkl` |
| 5 | [scripts/build_test_sets.py](scripts/build_test_sets.py) | `test-sets` | Corre el modelo final de cada nivel sobre su test set y agrega las métricas de error. | `artifacts/test_sets/{level}.parquet`, `output/test_metrics.json` |

Métricas: WAPE, Bias, WRMSSE (y por serie también MASE/SPEC) — ver [src/evaluation/](src/evaluation/).

## Estructura

```
backup/     Zip de Kaggle descargado por create-database (no versionado, ver Configuración)
data/       raw/ (CSV de M5), processed/ (por nivel, sin features), datasets/ (con features, listo para entrenar), m5.db — solo se versiona la estructura
config/     Parámetros del proyecto separados por responsabilidad (paths, features, levels, model)
queries/    SQL para crear la base DuckDB y los datasets base
src/        Lógica reutilizable (datos, features, modelado, evaluación)
scripts/    Puntos de entrada ejecutables (orquestan src/)
notebooks/  Exploración (eda) y prototipado del modelo (model)
artifacts/  results/, plots/, models/, test_sets/ y optuna_study.db — generados por train-dataset y test-sets
output/     test_metrics.json — métricas del modelo final sobre test (test-sets)
```
