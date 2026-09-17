# forecast-tfm

## Requisitos

- [uv](https://docs.astral.sh/uv/) (Python 3.13, ver `.python-version`)
- Opcional: Cuenta de [Kaggle](https://www.kaggle.com/) (solo para descargar los datos, ya que sus condiciones de uso no permiten publicarlos).

```bash
git clone git@github.com:njimenez94/forecast-tfm.git && cd forecast-tfm
uv sync
```

## 1. Conseguir los datos

**Opción A — con API de Kaggle**

```bash
export KAGGLE_API_TOKEN=token
```

Posteriormente al ejecutar código, se usará la key.

**Opción B — manual, sin API key**

1. Entrar a [la página de datos de Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) descargar el zip.
2. Dejarlo sin descomprimir en `backup/m5-forecasting-accuracy.zip`

## 2. Correr el pipeline

**1) Crear base de datos**

```bash
make create-database
```

Crea `data/m5.db` a partir del zip de Kaggle.

**2) Procesar datos**

```bash
make process-data
```

Genera `data/processed/{daily,weekly}/level_*.parquet`.

**3) Construir datasets**

```bash
make build-datasets
```

Arma `data/datasets/{daily,weekly}/dataset_level_*.parquet` con features derivadas.

**4) Entrenar modelo**

```bash
make train-dataset
```

Entrena y evalúa, guarda en `artifacts/{results,plots,models}/`.

**5) Evaluar en test**

```bash
make test-sets
```

Evalúa el modelo final vs. test, guarda en `artifacts/test_sets/` y `output/test_metrics.json`.

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

## 3. Config por defecto

El pipeline ya viene configurado con lo que uso para el TFM (`config/`); por defecto corre así, sin tocar nada:

- **Niveles activos** (`config.levels.ACTIVE_LEVEL_IDS`): los 12 niveles M5, en daily y weekly. Los niveles item-level (10-12) vienen acotados a `dept_id=FOODS_3` (y `state_id=CA`/`store_id=CA_3` en 11/12) porque sin filtro son inviables de materializar.
- **Perfil de entrenamiento** (`config.training`): `efficient` — todas las fases (baselines, feature selection, SHAP, Optuna) con presupuesto recortado para poder correr los 12 niveles x 2 grains en una sola pasada.
- **Modelo** (`config.model`): LightGBM, objective `regression_l2` (o `tweedie` en niveles 11-12 por la cantidad de ceros).

Para cambiar esto sin editar `config/`, usar `ARGS` al correr los comandos de la sección anterior, por ejemplo `ARGS="--levels 1,9,12 --profile fast"`. Para un cambio permanente, editar el archivo correspondiente en `config/`.

## 4. Servir el modelo (API)

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
backup/     zip de Kaggle
data/       raw/ (CSV de M5), processed/ (por nivel), datasets/ (con features)
config/     parámetros del proyecto (paths, features, levels, model, training)
queries/    SQL para crear la base DuckDB
api/        API de predicción sobre los artifacts entrenados
src/        lógica reutilizable (datos, features, modelado, entrenamiento, evaluación)
scripts/    puntos de entrada ejecutables (orquestan src/)
notebooks/  exploración y prototipado
docs/       archivos varios .md mayormente
artifacts/  results/, plots/, models/, test_sets/ -- generados por train-dataset y test-sets
logs/       un .log por corrida, por script
```
