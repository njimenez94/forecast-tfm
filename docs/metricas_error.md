# Métricas de error

Documento de referencia sobre las métricas de error usadas en el pipeline de forecasting (`src/evaluation/`), pensado para justificar en el informe por qué se reporta más de una métrica y qué mide cada una. Todos los ejemplos numéricos de este documento son **sintéticos** (series inventadas de pocos días, para que la cuenta se siga a mano) y fueron verificados ejecutando las funciones reales del proyecto — no son resultados del modelo entrenado.

## 1. Por qué varias métricas y no una sola

El dataset M5 tiene dos particularidades que ninguna métrica única captura por completo:

1. **Demanda intermitente**: a nivel item-tienda (L10-L12) muchas series tienen más días con venta 0 que con venta positiva. Métricas basadas en error porcentual (MAPE, SMAPE) se disparan o quedan indefinidas cuando la venta real es 0.
2. **Jerarquía de escalas**: el proyecto entrena y evalúa 12 niveles de agregación distintos (`config/levels.py`), desde ventas totales (L1) hasta item-tienda (L12). Una serie de miles de unidades/día y una serie intermitente de 0-3 unidades/día no son comparables con un error absoluto (MAE, RMSE) crudo.

Por eso se combinan métricas **puntuales** (por fila/día, fáciles de leer) con métricas **escaladas** (por serie, comparables entre niveles porque se normalizan contra un benchmark naive calculado sobre esa misma serie).

## 2. Notación

- $y_t$: venta real en el período $t$ (día o semana, según el nivel).
- $\hat{y}_t$: predicción del modelo para $t$.
- $n$: cantidad de períodos evaluados (el horizonte de validación/test — 28 días o 4 semanas según `config.VALID_PERIODS`/`TEST_PERIODS`).
- $i$: índice de serie (una combinación item/tienda/etc. según el nivel) cuando la métrica se calcula "por serie" antes de agregarse.

## 3. Métricas puntuales (`src/evaluation/point.py`)

Se calculan sobre el conjunto de filas evaluado (todas las series y fechas del horizonte juntas), sin distinguir de qué serie viene cada fila.

### Ejemplo compartido

10 días de una combinación item-tienda:

| día | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| $y_t$ (real) | 0 | 2 | 0 | 3 | 1 | 0 | 4 | 0 | 2 | 3 |
| $\hat{y}_t$ (predicho) | 1 | 1 | 1 | 2 | 2 | 1 | 3 | 1 | 2 | 2 |

### 3.1 WAPE (Weighted Absolute Percentage Error)

**Qué mide:** el error absoluto total como fracción de la venta total. Es la métrica "resumen" más fácil de comunicar (se lee como un porcentaje) y no se rompe con ceros individuales, porque no divide fila a fila sino que agrega numerador y denominador por separado.

$$\text{WAPE} = \frac{\sum_t |y_t - \hat{y}_t|}{\sum_t |y_t|}$$

**Ejemplo:** $\sum |y_t - \hat y_t| = 9$, $\sum y_t = 15$ → **WAPE = 9/15 = 0.60 (60%)**.

**Interpretación:** el modelo se equivoca, en total, por un 60% de la venta real acumulada. Cuanto más cerca de 0, mejor.

**Uso en el proyecto:** métrica principal de lectura rápida en la comparación de modelos ([src/evaluation/reports.py:31](../src/evaluation/reports.py#L31)) y en el detalle por serie ([reports.py:142](../src/evaluation/reports.py#L142)).

### 3.2 Bias

**Qué mide:** si el modelo sobreestima o subestima, también normalizado por la venta total (mismo denominador que WAPE, para que ambas se puedan comparar directamente).

$$\text{Bias} = \frac{\sum_t (\hat{y}_t - y_t)}{\sum_t |y_t|}$$

**Ejemplo:** $\sum \hat y_t = 16$, $\sum y_t = 15$ → **Bias = 1/15 ≈ +0.067 (+6.7%)**.

**Interpretación:** signo positivo = el modelo sobreestima (predice de más); signo negativo = subestima. Un WAPE bajo con Bias alto indica que el modelo acierta la magnitud promedio pero está sesgado en una dirección — señal útil para decidir si conviene un ajuste de calibración.

**Uso en el proyecto:** [point.py:10-12](../src/evaluation/point.py#L10-L12), reportado junto a WAPE/WRMSSE en cada comparación de modelos.

### 3.3 MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{n}\sum_t |y_t - \hat{y}_t|$$

**Ejemplo:** MAE = 9/10 = **0.9 unidades/día** de error promedio.

**Interpretación:** mismo tipo de información que WAPE pero en unidades de venta (no normalizado), por lo que solo es comparable entre series de escala similar — por eso no se usa para comparar niveles distintos.

### 3.4 RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_t (y_t - \hat{y}_t)^2}$$

**Ejemplo:** RMSE = $\sqrt{9/10}$ ≈ **0.949**.

**Interpretación:** como eleva el error al cuadrado, penaliza más los errores grandes que el MAE (dos errores de 1 pesan igual que MAE, pero un error de 2 pesa como cuatro errores de 1). Es también el objetivo de entrenamiento (`objective="regression_l2"`) para la mayoría de los niveles — ver §5.

### 3.5 SMAPE

$$\text{SMAPE} = \frac{1}{n}\sum_t \frac{2\,|\hat{y}_t - y_t|}{|y_t| + |\hat{y}_t| + \varepsilon}$$

**Ejemplo:** **SMAPE ≈ 1.042 (104%)**, a pesar de que WAPE es "solo" 60%.

**Interpretación — por qué se reporta pero no se usa como métrica principal:** SMAPE promedia un ratio *por fila*, así que cada día con venta real 0 (frecuente en este dataset) contribuye un término cercano al máximo (2) apenas el modelo predice algo distinto de 0, sin importar cuán chica sea la predicción. Con demanda intermitente esto infla el promedio y lo hace difícil de interpretar — es la razón concreta por la que WAPE (que agrega antes de dividir) es la métrica de lectura rápida en este proyecto y SMAPE queda como referencia secundaria.

### 3.6 RMSLE (Root Mean Squared Log Error)

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_t \big(\log(1+\hat{y}_t) - \log(1+y_t)\big)^2}$$

**Ejemplo:** **RMSLE ≈ 0.497**.

**Interpretación:** al trabajar en escala logarítmica, un mismo error absoluto pesa menos cuanto más alta es la venta real (penaliza más subestimar que sobreestimar). Útil para comparar el error relativo entre series con picos altos y series casi planas sin que las primeras dominen el promedio, algo que RMSE sí haría.

### 3.7 Tracking Signal

$$\text{TS} = \frac{\sum_t (y_t - \hat{y}_t)}{\text{MAD}}, \quad \text{MAD} = \frac{1}{n}\sum_t |y_t - \hat{y}_t|$$

**Ejemplo:** $\sum(y_t-\hat y_t) = -1$, MAD = 0.9 → **TS ≈ -1.11**.

**Interpretación:** métrica clásica de demand forecasting (Brown, 1959) para detectar sesgo *sostenido* en el tiempo, no solo su magnitud promedio (eso ya lo cubre Bias). Valores dentro de aproximadamente ±4 se consideran normales; fuera de ese rango indican que el error no es ruido sino un sesgo sistemático que conviene corregir. En el ejemplo, -1.11 está dentro de rango normal.

## 4. Métricas escaladas y jerárquicas (`src/evaluation/scaled.py`)

Estas métricas se calculan **por serie** contra un benchmark *naive* (repetir el valor de hace $m$ períodos) calculado *in-sample* sobre el propio train de esa serie, y después se agregan (promedio simple o ponderado por venta en $). Esto es lo que las hace comparables entre niveles de escala muy distinta: un RMSSE de 0.8 en L1 (ventas totales) y un RMSSE de 0.8 en L12 (item-tienda) significan lo mismo — "20% mejor que repetir el día anterior" — aunque las unidades de venta sean completamente distintas.

### Ejemplo compartido

Historia de train de una serie S1 (9 días, para calcular la escala naive *in-sample*) y horizonte de validación evaluado (5 días):

| train | día 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| $y_t$ | 1 | 0 | 2 | 0 | 3 | 1 | 0 | 2 | 1 |

| valid | día 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|
| $y_t$ (real) | 2 | 0 | 3 | 1 | 4 |
| $\hat y_t$ (predicho) | 1 | 1 | 2 | 2 | 3 |

### 4.1 RMSSE (Root Mean Squared Scaled Error)

**Qué mide:** RMSE del horizonte evaluado, escalado por el RMSE del naive de $m$ pasos calculado in-sample sobre train (mismo tipo de escala que usó la competencia M5).

$$\text{RMSSE}_i = \sqrt{\dfrac{\dfrac{1}{n}\sum_{t}(y_t - \hat y_t)^2}{\dfrac{1}{T-m}\sum_{t=m+1}^{T}(y_t - y_{t-m})^2}}$$

donde el denominador se calcula sobre el **train** de la serie ($T$ = días de train, $m=1$ para el target `sales`; $m=N$ para el target acumulado `cumN`, comparando cada ventana de N períodos con la anterior no solapada — ver `compute_naive_scales`).

**Ejemplo:**
- Escala naive in-sample (denominador): diffs de train = $[-1,2,-2,3,-2,-1,2,-1]$ → MSE = 3.5.
- Error del modelo en valid (numerador): errores $[1,-1,1,-1,1]$ → MSE = 1.
- **RMSSE = √(1 / 3.5) ≈ 0.535**.

**Interpretación:** RMSSE < 1 significa que el modelo comete, en términos de esa escala cuadrática, menos error que "predecir el mismo valor de ayer" sobre el propio historial de la serie. RMSSE = 1 es empatar al naive; > 1 es perder contra él.

**Uso en el proyecto:** [scaled.py:72-81](../src/evaluation/scaled.py#L72-L81); antes de calcularlo se aplica `clip_closed_stores` para forzar a 0 la predicción en fechas de cierre de tienda conocido por calendario (evita castigar al modelo por algo que no podía predecir con datos, ver [scaled.py:196-203](../src/evaluation/scaled.py#L196-L203)).

### 4.2 WRMSSE (Weighted RMSSE)

**Qué mide:** el RMSSE de cada serie ($\text{RMSSE}_i$), agregado como promedio ponderado por el peso $w_i$ = ventas en \$ de esa serie en los últimos días de train (`compute_weights`, ventana `last_days=28` por defecto).

$$\text{WRMSSE} = \sum_i \frac{w_i}{\sum_j w_j}\,\text{RMSSE}_i$$

**Ejemplo** (agregando una segunda serie S2, con menos unidades vendidas pero precio más alto — ventana de pesos acortada a 9 días solo para que quepa en el ejemplo):

| serie | ventas train (u.) | precio | peso \$ | RMSSE |
|---|---|---|---|---|
| S1 | 10 | 10 | 100 | 0.535 |
| S2 | 4 | 40 | 160 | 0.505 |

**WRMSSE = (100·0.535 + 160·0.505) / 260 ≈ 0.516**

**Interpretación:** aunque S2 vendió menos unidades, pesa más en el WRMSSE final porque su precio la hace más relevante en \$ — el error en las series que más facturan importa más que el error en las que casi no se venden. Es la métrica que la competencia M5 original usa para rankear modelos, y la que este proyecto usa como **eval metric de LightGBM durante Optuna/entrenamiento** (no para decidir gradientes, solo para trackear/rankear — el objetivo real de boosting es RMSE o Tweedie deviance, ver §5).

**Uso en el proyecto:** [scaled.py:84-95](../src/evaluation/scaled.py#L84-L95); factory para LightGBM en [objectives.py:36-57](../src/evaluation/objectives.py#L36-L57).

### 4.3 MASE y WMASE (Mean Absolute Scaled Error)

Igual que RMSSE/WRMSSE pero con MAE en vez de RMSE (Hyndman & Koehler, 2006) — penaliza menos los errores grandes puntuales, más robusta a outliers.

$$\text{MASE}_i = \dfrac{\dfrac{1}{n}\sum_t |y_t - \hat y_t|}{\dfrac{1}{T-m}\sum_{t=m+1}^{T}|y_t - y_{t-m}|}, \qquad \text{WMASE} = \sum_i \frac{w_i}{\sum_j w_j}\,\text{MASE}_i$$

**Ejemplo** (mismos datos que 4.1/4.2): escala naive (MAE in-sample) = 1.75; MAE del modelo en valid = 1 → **MASE = 1/1.75 ≈ 0.571**. Con la segunda serie del ejemplo anterior, **WMASE ≈ 0.466**.

**Uso en el proyecto:** [scaled.py:135-176](../src/evaluation/scaled.py#L135-L176), reportada junto a WRMSSE en la comparación de modelos y en el detalle por serie.

### 4.4 SPEC (Stock-keeping-oriented Prediction Error Costs)

**Qué mide:** a diferencia de las métricas anteriores, SPEC no solo mira la *magnitud* del error sino **cuándo** ocurre, con un costo asimétrico pensado para gestión de inventario (Martin, Hewamalage & Bergmeir, [arXiv:2004.10537](https://arxiv.org/abs/2004.10537)):

- predecir de más un día antes de que llegue la demanda real ⇒ costo de **sobre-stock** (mantener inventario de más), ponderado por $a_2$.
- predecir de menos, quedándose corto cuando sí llegó la demanda ⇒ costo de **quiebre de stock** (demanda no satisfecha), ponderado por $a_1$.

En este proyecto $a_1=0.75$, $a_2=0.25$: el quiebre de stock se penaliza 3 veces más que el sobre-stock, reflejando que en retail suele doler más no tener el producto que tenerlo de más.

**Ejemplo:** un pico de demanda real de 3 unidades el día 2, con tres forecasts distintos que tienen el **mismo error absoluto total** (3 unidades erradas):

| forecast | día 1 | día 2 | día 3 | día 4 | SPEC |
|---|---|---|---|---|---|
| $y_t$ (real) | 0 | 3 | 0 | 0 | — |
| perfecto | 0 | 3 | 0 | 0 | **0.000** |
| adelantado (llega 1 día antes → sobra stock 1 día) | 3 | 0 | 0 | 0 | **0.188** |
| atrasado (llega 1 día tarde → falta stock 1 día) | 0 | 0 | 3 | 0 | **0.563** |

**Interpretación:** WAPE/RMSE tratarían "adelantado" y "atrasado" como igual de malos (mismo error absoluto). SPEC no: el atrasado cuesta exactamente 3 veces más que el adelantado (0.563/0.188 = 3 = $a_1/a_2$), porque llegar tarde significa quiebre de stock. Es la métrica más alineada con el objetivo de negocio (reponer inventario), pero no se usa como objetivo de entrenamiento por ser costosa de derivar (no es diferenciable de forma directa para LightGBM) — se calcula solo para reportar.

**Uso en el proyecto:** [scaled.py:10-41](../src/evaluation/scaled.py#L10-L41) (implementación vectorizada con sumas acumuladas), agregada por serie igual que WRMSSE en [scaled.py:98-115](../src/evaluation/scaled.py#L98-L115).

## 5. Objetivo de entrenamiento: RMSE vs. Tweedie deviance

Todas las métricas anteriores son de **evaluación** (miden qué tan bueno es un modelo ya entrenado). El **objetivo de entrenamiento** de LightGBM — lo que realmente guía los gradientes durante el boosting — es distinto y depende del nivel (`config/model.py`, `TWEEDIE_LEVELS = {11, 12}`):

- **Niveles 1-10** (más agregados, pocos ceros): `objective="regression_l2"`, equivalente a minimizar RMSE.
- **Niveles 11-12** (item-tienda, la mayoría de los días en 0): `objective="tweedie"`.

**Por qué Tweedie para los niveles más granulares:** RMSE no distingue *dónde* ocurre el error en relación a la escala del valor real, solo su magnitud absoluta. La deviance de Tweedie sí — penaliza el error en términos relativos a la media esperada, lo cual importa mucho en series con casi todo ceros y algún pico ocasional.

**Ejemplo** — dos predicciones con **exactamente el mismo RMSE**, un pico real de 10 el día 5:

| forecast | día 1 | 2 | 3 | 4 | 5 | RMSE | Tweedie deviance (power=1.5) |
|---|---|---|---|---|---|---|---|
| real | 0 | 0 | 0 | 0 | 10 | — | — |
| error repartido en los ceros | 1 | 1 | 1 | 1 | 10 | 0.894 | **3.200** |
| mismo error, concentrado en el pico | 0 | 0 | 0 | 0 | 8 | 0.894 | **0.035** |

**Interpretación:** con RMSE ambos forecasts son idénticos de "malos". Con Tweedie, equivocarse por 1 unidad en un día que debía ser 0 cuesta casi 100 veces más que equivocarse por 2 unidades en el día del pico de 10. Esto es justo lo que se necesita en L11/L12: si se entrenara con RMSE puro, el modelo aprendería que la forma más barata de reducir el error es predecir siempre un valor chico constante (casi nunca hay picos grandes, así que casi nunca se paga el error grande); con Tweedie, acertar los ceros pesa proporcionalmente y el modelo no puede "regalarse" esos días.

**Uso en el proyecto:** [config/model.py:8](../config/model.py#L8) (qué niveles usan Tweedie), [src/evaluation/objectives.py:9-18](../src/evaluation/objectives.py#L9-L18) (`objective_metric`, la métrica coherente con el objective de cada nivel — decide selección de features y ranking de Optuna; WRMSSE se calcula aparte solo para informar, no para decidir).

## 6. Resumen: qué métrica mirar y dónde aparece

| Métrica | Tipo | Qué responde | Dónde se calcula/usa |
|---|---|---|---|
| WAPE | puntual | "¿Qué % de la venta total erré?" | lectura rápida en toda comparación de modelos |
| Bias | puntual | "¿Sobreestimo o subestimo?" | idem, junto a WAPE |
| MAE / RMSE | puntual | error crudo en unidades de venta | referencia interna, no comparable entre niveles |
| SMAPE | puntual | error % simétrico por fila | secundaria — se infla con demanda intermitente |
| RMSLE | puntual | error relativo, amortigua picos | comparar series con escalas muy distintas |
| Tracking Signal | puntual | ¿el sesgo es sistemático en el tiempo? | detección de deriva/sesgo sostenido |
| RMSSE / WRMSSE | escalada | "¿mejor que repetir ayer?", ponderado por \$ venta | eval metric de LightGBM en Optuna; métrica de ranking estilo M5 |
| MASE / WMASE | escalada | igual que RMSSE/WRMSSE, más robusta a outliers | reportes de comparación y detalle por serie |
| SPEC | escalada, asimétrica | costo de inventario por *cuándo* se erró | solo reporte — no es objetivo de entrenamiento |
| RMSE / Tweedie deviance | objetivo de entrenamiento | qué minimiza LightGBM durante el boosting | `config.lgbm_params()` según nivel |

Todas las métricas de evaluación (puntuales y escaladas) se calculan juntas por `evaluate_predictions` ([reports.py:14-50](../src/evaluation/reports.py#L14-L50)) para la comparación de modelos, y por `build_series_metrics` ([reports.py:88-151](../src/evaluation/reports.py#L88-L151)) para el detalle por serie — ambas quedan en `artifacts/results/*_model_comparison.csv` y `*_series_metrics.csv` (ver [scripts/train_dataset.py](../scripts/train_dataset.py)).
