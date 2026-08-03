# forecast-tfm

Pipeline de forecasting de demanda (dataset **M5**) con LightGBM y validación temporal.

El flujo es: descomprimir los datos → construir la base de datos → generar features → comparar/tunear modelos → entrenar el modelo final → predecir sobre validación.

## Estructura

```
doc/        Datos crudos comprimidos (m5-forecasting-accuracy.zip)
data/       raw/ (CSV de M5) y processed/ (dataset.parquet por nivel, sin features derivadas) — solo se versiona la estructura
config/     Parámetros del proyecto separados por responsabilidad (paths, features, model)
queries/    SQL para crear la base DuckDB y el dataset base
src/        Lógica reutilizable (datos, features, modelado, evaluación)
scripts/    Puntos de entrada ejecutables (orquestan src/)
notebooks/  Exploración (eda) y prototipado del modelo (model)
artifacts/  Salidas generadas: datasets/ (datasets listos para modelos), models/, forecast
```

## Scripts

Cada script vive en [scripts/](scripts/) y se ejecuta como módulo (`uv run python -m scripts.<nombre>`) o con su target de `make`. Todos comparten el mismo preámbulo: **cargar dataset base → construir features → split temporal**, definido en [scripts/_common.py](scripts/_common.py).

| Script | `make` | Qué hace |
|--------|--------|----------|
| [scripts/create_database.py](scripts/create_database.py) | `make create_database` | Inicializa todo de una vez: descomprime `doc/m5-forecasting-accuracy.zip` en `data/raw/` y crea la base DuckDB `data/m5.db`. Es el primer paso del pipeline. |
| [scripts/_common.py](scripts/_common.py) | — | Helper compartido (`prepare_data`): carga el dataset base desde DuckDB, construye features y hace el split temporal train/valid. No es ejecutable por sí solo; lo usan los demás scripts. |
| [scripts/search.py](scripts/search.py) | `make search` | Compara varios modelos con validación temporal e imprime el ranking ordenado por WAPE. Sirve para elegir el modelo base antes de tunear. |
| [scripts/tune.py](scripts/tune.py) | `make tune N=50` | Optimiza los hiperparámetros de LightGBM con Optuna (`N` trials, 50 por defecto) y guarda los mejores en `artifacts/best_params.json`. |
| [scripts/train.py](scripts/train.py) | `make train` | Entrena el modelo final y lo guarda en `artifacts/model.txt`. Usa `artifacts/best_params.json` si existe (salida de `tune`); si no, los params por defecto de [config/](config/). Reporta WAPE y Bias en validación. |
| [scripts/predict.py](scripts/predict.py) | `make predict` | Carga `artifacts/model.txt`, predice sobre validación, guarda el forecast en `artifacts/forecast.parquet` e imprime el top 10 de `agg_id` por WAPE. Requiere haber entrenado antes. |

## Uso típico

```bash
make create_database   # 1. descomprime el zip de M5 y crea data/m5.db
make search            # 2. (opcional) compara modelos
make tune N=50         # 3. (opcional) tunea hiperparámetros -> best_params.json
make pipeline          # 4. train + predict en un paso
```

`make pipeline` equivale a `make train && make predict`. Para empezar de cero usa `make clean` (borra `artifacts/`).

Ejecuta `make help` para ver todos los targets disponibles.

## Artefactos generados

| Archivo | Generado por | Contenido |
|---------|--------------|-----------|
| `artifacts/best_params.json` | `tune` | Mejores hiperparámetros encontrados por Optuna |
| `artifacts/model.txt` | `train` | Modelo LightGBM entrenado |
| `artifacts/forecast.parquet` | `predict` | Predicciones sobre el set de validación |
