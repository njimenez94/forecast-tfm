# forecast-tfm

Forecasting de demanda sobre el dataset **M5** (LightGBM vía mlforecast, validación temporal). La metodología, los resultados y las decisiones de diseño están en el informe del TFM y en el video explicativo — este README es la guía práctica para correr el proyecto.

## Requisitos

- [uv](https://docs.astral.sh/uv/) (Python 3.13, ver `.python-version`)
- Cuenta de Kaggle que haya aceptado las reglas de [m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) — solo para el paso 1

```bash
git clone <repo> && cd forecast-tfm
uv sync
```

## 1. Conseguir los datos

El pipeline necesita 3 CSV de la competencia: `calendar.csv`, `sell_prices.csv` y `sales_train_evaluation.csv`.

**Opción A — con API de Kaggle**

```bash
export KAGGLE_API_TOKEN=tu_token   # generado en https://www.kaggle.com/settings/api
make create-database
```
(alternativa a la variable de entorno: `uv run kaggle auth login`, autentica por navegador).

`create-database` descarga el zip a `backup/m5-forecasting-accuracy.zip`, lo descomprime en `data/raw/` y arma la base DuckDB en `data/m5.db`.

**Opción B — manual, sin API key**

1. Entrá logueado a la [página de datos de la competencia](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) (hay que haber aceptado las reglas) y descargá el zip con "Download All".
2. Dejalo sin descomprimir en `backup/m5-forecasting-accuracy.zip` (creá la carpeta) y corré `make create-database` — lo encuentra y salta la descarga.
   - O descomprimilo vos mismo y copiá los 3 CSV directo en `data/raw/`; `make create-database` solo descarga/descomprime si no los encuentra, así que si ya están ahí pasa directo a crear la base.

## 2. Correr el pipeline

| Paso | Comando | Entrada → salida |
|---|---|---|
| 1 | `make create-database` | zip de Kaggle → `data/m5.db` |
| 2 | `make process-data` | `data/m5.db` → `data/processed/{daily,weekly}/level_*.parquet` |
| 3 | `make build-datasets` | + features derivadas → `data/datasets/{daily,weekly}/dataset_level_*.parquet` |
| 4 | `make train-dataset` | entrena y evalúa → `artifacts/{results,plots,models}/` |
| 5 | `make test-sets` | modelo final vs. test → `artifacts/test_sets/`, `output/test_metrics.json` |

`make pipeline` encadena 2-4. Todo usa los defaults de `config/` (niveles activos en `config.ACTIVE_LEVEL_IDS`); para acotar: `ARGS="--levels 1,9,12"`, `ARGS="--profile fast"` (ver perfiles abajo), combinables: `ARGS="--levels 1 --profile fast"`.

Para verificar que un cambio en `src/` no rompió nada, rápido y sin pisar los artifacts ya entrenados:

```bash
make smoke-check   # perfil fast, nivel 1, no guarda artifact
```

### Perfiles de entrenamiento (`--profile`)

Por defecto `train-dataset` entrena los 12 niveles x 2 grains con `efficient`, y en los niveles grandes puede tardar (bench de 5 familias + Optuna + SHAP por combinación). Para una primera corrida, `--profile fast` (sumale `--levels 1` para acotar más):

| Perfil | Qué hace | Cuándo usarlo |
|---|---|---|
| `fast` | Un solo split, sin Optuna/SHAP/feature selection | Probar que el pipeline corre de punta a punta |
| `moderate` | Optuna recortado, fit final completo | Primera versión entrenable |
| `efficient` (default) | Todas las fases, presupuesto recortado | Corrida completa en tiempo razonable |
| `optimized` | Pipeline completo, presupuesto original | Métricas finales del TFM |

Corridas largas conviene dejarlas en `tmux`/`nohup`: el progreso queda en `logs/train_dataset/` igual si se corta la terminal.

## 3. Servir el modelo (API)

`api/` sirve los artifacts ya entrenados por HTTP — no recalcula features, recibe el vector ya procesado y corre `model.predict()`.

```bash
make serve-api                                        # uvicorn --reload en :8000
make docker-api                                        # build + contenedor en :8000
make testing-api ARGS="--level 9 --grain daily --n 3"  # prueba contra la API ya corriendo
```

Endpoints: `GET /health`, `GET /levels`, `POST /predict/{level_id}`. Parámetros y detalle de nivel/grain en `api/main.py` y `api/registry.py`.

## Tests

```bash
make test       # rápido: schema/features/API, ~10s, sin datos de Kaggle
make test-all   # + harness de reproducibilidad (entrena modelos sobre datos sintéticos)
```

Antes de mergear un cambio en `src/features/` o `src/modeling/`, además de `make test`:

```bash
make train-dataset ARGS="--levels 1 --profile fast"
make check-regression ARGS="--level 1"   # falla si wape/wrmsse empeoró >5% vs. la versión anterior en registry.json
```

## Estructura

```
backup/     zip de Kaggle (no versionado, ver paso 1)
data/       raw/ (CSV de M5), processed/ (por nivel), datasets/ (con features), m5.db -- solo se versiona la estructura de carpetas
config/     parámetros del proyecto (paths, features, levels, model, training)
queries/    SQL para crear la base DuckDB
api/        API de predicción sobre los artifacts entrenados
src/        lógica reutilizable (datos, features, modelado, entrenamiento, evaluación)
scripts/    puntos de entrada ejecutables (orquestan src/)
notebooks/  exploración y prototipado
report/     notebook de resultados finales para el informe
docs/       informe del TFM y notas de metodología
artifacts/  results/, plots/, models/, test_sets/ -- generados por train-dataset y test-sets
output/     test_metrics.json
logs/       un .log por corrida, por script
```

**Sobre `artifacts/`:** `artifacts/models/*.pkl` sí está versionado (vía Git LFS) porque es liviano — modelo + features + métricas, sin datos — y así el repo se puede usar sin reentrenar desde cero. El resto de `artifacts/` (results, plots, optuna_study.db, test_sets) no se versiona porque se reproduce corriendo el pipeline. `make clean` borra `data/processed`, `data/datasets` y todo `artifacts/`.
