# forecast-tfm

Pipeline de forecasting de demanda (dataset **M5**) con LightGBM (vía mlforecast) y validación temporal.

El flujo es: descomprimir los datos → crear la base de datos → construir datasets por nivel de agregación → generar features derivadas → entrenar y evaluar por nivel.

## Estructura

```
backup/     Datos crudos comprimidos (m5-forecasting-accuracy.zip)
data/       raw/ (CSV de M5, generado) y processed/ (dataset por nivel, sin features derivadas) — solo se versiona la estructura
config/     Parámetros del proyecto separados por responsabilidad (paths, features, levels, model)
queries/    SQL para crear la base DuckDB y los datasets base
src/        Lógica reutilizable (datos, features, modelado, evaluación)
scripts/    Puntos de entrada ejecutables (orquestan src/)
notebooks/  Exploración (eda) y prototipado del modelo (model)
artifacts/  Salidas generadas: datasets/ (features listas para entrenar) y models/ (modelos por nivel/target)
output/     Métricas de los experimentos de entrenamiento (experiment_results.parquet)
```

## Scripts

Cada script vive en [scripts/](scripts/) y se ejecuta como módulo (`uv run python -m scripts.<nombre>`) o con su target de `make`.

| Script | `make` | Qué hace |
|--------|--------|----------|
| [scripts/create_database.py](scripts/create_database.py) | `make create-database` | Descomprime `backup/m5-forecasting-accuracy.zip` en `data/raw/` y crea la base DuckDB `data/m5.db`. Es el primer paso del pipeline. |
| [scripts/process_data.py](scripts/process_data.py) | `make process-data` | Genera, por nivel de agregación (L1 → L12), un parquet en `data/processed/` con las ventas y los targets acumulados (`cumN`). Acepta `ARGS="--levels 1,9,12"` para limitar a ciertos niveles y `ARGS="--counts"` para solo contar series/filas. |
| [scripts/build_datasets.py](scripts/build_datasets.py) | `make build-datasets` | Aplica `src.features.engineer` sobre cada parquet de `data/processed/` (lags, rolling, momentum) y guarda el dataset final en `artifacts/datasets/`. Acepta `ARGS="--levels 1,9,12"`. |
| [scripts/train_datasets.py](scripts/train_datasets.py) | `make train-datasets` | Entrena un MLForecast por nivel/grain/target sobre `artifacts/datasets/`, evalúa en valid y test (WAPE, Bias, WRMSSE), guarda los modelos en `artifacts/models/` y las métricas en `output/experiment_results.parquet`. |

## Uso típico

```bash
make create-database   # 1. descomprime el zip de M5 y crea data/m5.db
make process-data       # 2. datasets base por nivel -> data/processed/
make build-datasets     # 3. features derivadas -> artifacts/datasets/
make train-datasets     # 4. entrena y evalúa por nivel -> artifacts/models/, output/experiment_results.parquet
```

También existe `make data-pipe`, que encadena los tres primeros pasos (`create-database process-data build-datasets`) en un solo comando. Cualquier target que acepte argumentos se invoca con `ARGS`, p.ej. `make process-data ARGS="--levels 1,9,12"`.

## Artefactos generados

| Archivo | Generado por | Contenido |
|---------|--------------|-----------|
| `data/m5.db` | `create-database` | Base de datos DuckDB con las tablas crudas de M5 |
| `data/processed/{daily,weekly}/level_*.parquet` | `process-data` | Dataset base por nivel de agregación, sin features derivadas |
| `artifacts/datasets/dataset_level_*.parquet` | `build-datasets` | Dataset final por nivel, con features/lags/rolling ya materializados |
| `artifacts/models/` | `train-datasets` | Modelos MLForecast entrenados, uno por nivel/grain/target |
| `output/experiment_results.parquet` | `train-datasets` | Métricas (WAPE, Bias, WRMSSE) por nivel/grain/target/split/horizonte |
