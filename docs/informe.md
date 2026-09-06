# Trabajo Fin de Máster: Previsión de Demanda en Retail sobre el Dataset M5 (Walmart)

**Autor:** Nicolás Jiménez
**Email:** jd.nicolas1@gmail.com

> **Nota de formato:** este documento se traspasó desde el borrador en Word/PDF y se completó a partir del estado real del repositorio `forecast-tfm` (código, notebooks y artefactos generados por el pipeline). A lo largo del texto hay comentarios HTML —invisibles en la vista previa renderizada de Markdown, visibles al abrir el archivo como texto plano o en modo "source"— con sugerencias concretas de qué gráfico, imagen o número conviene insertar en cada punto al pasar esto de vuelta a Word. Leyenda:
> - `📊 SUGERENCIA GRÁFICO` — qué visualizar y de dónde sacarlo o regenerarlo.
> - `🔢 SUGERENCIA DATO` — qué número, tabla o cifra destacar.
> - `🖼️ IMAGEN` — imagen real del repo ya insertada, o sugerida para generar.
> - `⚠️ REVISAR ANTES DE ENTREGAR` — algo puntual en el código/notebook conviene resolverlo antes de mostrarlo en la defensa.

## Índice

1. [Introducción y contexto de negocio](#1-introducción-y-contexto-de-negocio)
2. [Descripción del dataset](#2-descripción-del-dataset)
   1. [Descripción nivel geográfico](#21-descripción-nivel-geográfico)
   2. [Descripción productos](#22-descripción-productos)
3. [Diseño experimental](#3-diseño-experimental)
4. [Análisis exploratorio](#4-análisis-exploratorio)
5. [Feature engineering](#5-feature-engineering)
6. [Experimentación y evaluación comparativa](#6-experimentación-y-evaluación-comparativa)
7. [Optimización de los modelos seleccionados](#7-optimización-de-los-modelos-seleccionados)
8. [Interpretabilidad](#8-interpretabilidad)
9. [Productivización vía API](#9-productivización-vía-api)
10. [Conclusiones y líneas futuras](#10-conclusiones-y-líneas-futuras)
11. [Bibliografía](#11-bibliografía)

---

## 1. Introducción y contexto de negocio

En las cadenas de retail, poder predecir cuál será la venta de unidades es un desafío que puede traer muchos beneficios: saber el nivel de venta que tendrá un producto en una sucursal con un grado de error controlado puede aportar múltiples beneficios en la planificación comercial y en la cadena de suministro.

El fin de este informe es desarrollar un modelo de predicción para el conjunto de datos de la competencia M5, equivalente a datos de Walmart durante el periodo 2011-2016, detallado a nivel de producto-sucursal-día.

A lo largo del proyecto se explora primero el dataset, se define el diseño experimental que se quiere lograr, se realiza la ingeniería de variables y la experimentación con sus resultados, para finalizar con la productivización vía API de estos modelos predictivos, de forma que puedan integrarse en sistemas productivos para los distintos usos de la compañía.

<!-- 🔢 SUGERENCIA DATO: para dar más gancho a la introducción, se puede anticipar aquí el resultado final (WAPE de test entre 6,3% y 7,5% según nivel, ver sección 7) como cifra de apertura. -->

## 2. Descripción del dataset

El dataset corresponde a la competencia de Kaggle llamada "M5 Forecasting – Accuracy", la cual contiene 5 archivos:

- **calendar.csv**: información de calendario; su llave es `d`, correspondiente al código del día, comenzando en `d_1` hasta `d_1969`. Incluye eventos, flags SNAP por estado y la semana `wm_yr_wk` usada para relacionar con los precios.
- **sales_train_evaluation.csv**: ventas diarias históricas en unidades por combinación item-tienda. Es el archivo efectivamente usado como fuente de ventas en este trabajo.
- **sales_train_validation.csv**: subconjunto de `sales_train_evaluation.csv` sin los últimos 28 días. No se utiliza en el pipeline actual, ya que `sales_train_evaluation.csv` contiene la totalidad de esta serie más los 28 días adicionales (no son conjuntos disjuntos).
- **sample_submission.csv**: plantilla de envío de la competencia Kaggle; no se usa como fuente de datos, solo como referencia de formato de la competencia original.
- **sell_prices.csv**: precio de venta por combinación item-tienda-semana (`wm_yr_wk`).

Se crea un proceso en el cual se extrae la información de todos estos archivos y se combinan para crear un dataset que se usará posteriormente y será el dataset base.

Este contiene cerca de 59 millones de registros (59.181.090 exactamente) que describen la venta de cada producto en cada sucursal a nivel diario desde 2011-01-29 hasta el 2016-05-22.

| series_id | state_id | store_id | cat_id | dept_id | item_id | date | sell_price | sales | mnt_gross_sales | snap | event_name | event_type |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FOODS_1_001_CA_1 | CA | CA_1 | FOODS | FOODS_1 | FOODS_1_001 | 2011-02-06 | 2,00 | 0 | 0,00 | 1 | SuperBowl | Sporting |
| FOODS_1_001_CA_1 | CA | CA_1 | FOODS | FOODS_1 | FOODS_1_001 | 2011-02-14 | 2,00 | 2 | 4,00 | 0 | ValentinesDay | Cultural |
| FOODS_1_001_CA_1 | CA | CA_1 | FOODS | FOODS_1 | FOODS_1_001 | 2011-02-21 | 2,00 | 0 | 0,00 | 0 | PresidentsDay | National |
| FOODS_1_001_CA_1 | CA | CA_1 | FOODS | FOODS_1 | FOODS_1_001 | 2011-03-09 | 2,00 | 2 | 4,00 | 1 | LentStart | Religious |
| FOODS_1_001_CA_1 | CA | CA_1 | FOODS | FOODS_1 | FOODS_1_001 | 2011-03-16 | 2,00 | 1 | 2,00 | 0 | LentWeek2 | Religious |

### 2.1 Descripción nivel geográfico

El dataset incluye la venta de 10 sucursales en 3 estados distintos. Tomando como referencia la venta de 2015, la distribución por estado es:

| Estado | % del total de ventas |
|---|---|
| CA | 44% |
| TX | 29% |
| WI | 27% |

<!-- 📊 SUGERENCIA GRÁFICO: barras horizontal "Distribución de ventas brutas por estado (2015)" — ya existía como imagen en el Word original; regenerar desde notebooks/01_eda.ipynb (o su versión notebooks/01_eda.py) filtrando por state_id. -->

Las tiendas dominantes se describen así:

| Tienda | % del total de ventas |
|---|---|
| CA_3 | 16,6% |
| CA_1 | 11,9% |
| TX_3 | 10,4% |
| WI_2 | 10,1% |
| TX_2 | 10,1% |
| CA_2 | 9,2% |
| WI_1 | 8,9% |
| TX_1 | 8,3% |
| WI_3 | 7,8% |
| CA_4 | 6,6% |

<!-- 📊 SUGERENCIA GRÁFICO: barras horizontal "Distribución de ventas brutas por tienda (2015)", mismo origen que el anterior. -->

### 2.2 Descripción productos

La jerarquía de producto sigue la estructura `cat_id → dept_id → item_id`: cada `item_id` pertenece a un `dept_id`, y cada `dept_id` pertenece a un `cat_id`. A continuación, un ejemplo con el primer producto de cada departamento:

| cat_id | dept_id | item_id |
|---|---|---|
| FOODS | FOODS_1 | FOODS_1_001 |
| FOODS | FOODS_2 | FOODS_2_001 |
| FOODS | FOODS_3 | FOODS_3_001 |
| HOBBIES | HOBBIES_1 | HOBBIES_1_001 |
| HOBBIES | HOBBIES_2 | HOBBIES_2_001 |
| HOUSEHOLD | HOUSEHOLD_1 | HOUSEHOLD_1_001 |
| HOUSEHOLD | HOUSEHOLD_2 | HOUSEHOLD_2_001 |

La venta por categoría (2015) se distribuye así:

| Categoría | % del total de ventas |
|---|---|
| FOODS | 55,2% |
| HOUSEHOLD | 30,9% |
| HOBBIES | 13,9% |

Y entre los 7 departamentos del dataset:

| Departamento | % del total de ventas |
|---|---|
| FOODS_3 | 34,5% |
| HOUSEHOLD_1 | 23,5% |
| FOODS_2 | 13,8% |
| HOBBIES_1 | 13,2% |
| HOUSEHOLD_2 | 7,5% |
| FOODS_1 | 6,9% |
| HOBBIES_2 | 0,7% |

Finalmente, la cantidad de productos (SKUs) se distribuye de la siguiente manera entre los distintos departamentos del dataset:

| Departamento | Cantidad de productos |
|---|---|
| FOODS_1 | 216 |
| FOODS_2 | 398 |
| FOODS_3 | 823 |
| HOBBIES_1 | 416 |
| HOBBIES_2 | 149 |
| HOUSEHOLD_1 | 532 |
| HOUSEHOLD_2 | 515 |

<!-- 📊 SUGERENCIA GRÁFICO: las 3 tablas anteriores tienen su versión de barras horizontal en el Word original (por categoría, por departamento en ventas y por departamento en nº de SKUs) — reutilizar esas imágenes o regenerarlas desde notebooks/01_eda.ipynb. -->

## 3. Diseño experimental

A partir del dataset anterior se generaron más de 190 variables derivadas de la base original, tales como *lags*, *rolling windows*, variables de calendario, entre otras (ver sección 5, "Feature engineering", para el detalle completo). Estas variables se someten a un algoritmo de selección de features que prioriza el método *backward*, evaluado mediante *permutation importance* (ver sección 7).

Una vez seleccionadas las variables, se entrenan en un modelo de Machine Learning cuyos hiperparámetros se optimizan con Optuna, el cual, a través de estadística bayesiana, busca la combinación de parámetros que minimice el error de la manera más eficiente posible dentro de un tiempo de búsqueda acotado.

Se exploran los 12 niveles de granularidad del dataset, descritos a continuación:

| Nivel | n_states | n_stores | n_cat | n_dept | n_items | n_series | n_days | n_rows |
|---|---|---|---|---|---|---|---|---|
| 1 - total | | | | | | 1 | 1.941 | 1.941 |
| 2 - state | 3 | | | | | 3 | 1.941 | 5.823 |
| 3 - cat | | | 3 | | | 3 | 1.941 | 5.823 |
| 4 - dept | | | 3 | 7 | | 7 | 1.941 | 13.587 |
| 5 - state_cat | 3 | | 3 | | | 9 | 1.941 | 17.469 |
| 6 - store | 3 | 10 | | | | 10 | 1.941 | 19.410 |
| 7 - state_dept | 3 | | 3 | 7 | | 21 | 1.941 | 40.761 |
| 8 - store_cat | 3 | 10 | 3 | | | 30 | 1.941 | 58.230 |
| 9 - store_dept | 3 | 10 | 3 | 7 | | 70 | 1.941 | 135.870 |
| 10 - item | | | 3 | 7 | 3.049 | 3.049 | 1.941 | 5.918.109 |
| 11 - item_state | 3 | | 3 | 7 | 3.049 | 9.147 | 1.941 | 17.754.327 |
| 12 - item_store | 3 | 10 | 3 | 7 | 3.049 | 30.490 | 1.941 | 59.181.090 |

<!-- 🔢 SUGERENCIA DATO: esta tabla ya está en el Word como tabla formal; conviene mantenerla con caption ("Tabla 1: niveles de agregación M5") y referenciarla por número desde el resto del informe, ya que se cita implícitamente en varias secciones posteriores. -->

El nivel 12 corresponde claramente al nivel más granular de la demanda y, por lo tanto, uno de los más desafiantes debido al volumen de datos que contiene. Para abordar esto, se construye un dataset por nivel; en el caso de los niveles más pesados, estos se segmentan según una o más jerarquías (por ejemplo, el nivel item-tienda se particiona por combinación tienda-departamento), con el fin de optimizar los recursos computacionales y evitar datasets de hasta 59 millones de registros, que pueden derivar en errores de Out of Memory (OOM).

Para mitigar este problema se utilizaron diversas herramientas orientadas a optimizar el uso de memoria, especialmente durante la etapa de creación de features. Estas se complementan con el uso tradicional de Pandas e incorporan tecnologías como Polars, que permite un procesamiento *lazy* y multi-hilo con un uso de memoria significativamente más eficiente que Pandas gracias a su motor en Rust, y DuckDB, que permite ejecutar consultas SQL analíticas directamente sobre archivos en disco (sin cargar el dataset completo en memoria), lo que resulta ideal para agregaciones y *joins* sobre volúmenes de datos que excederían la RAM disponible.

En la práctica, esto se tradujo en un límite explícito de memoria para el motor de base de datos, escritura directa a disco en formato Parquet sin materializar resultados intermedios en memoria, lectura diferida de esos archivos con reducción de precisión numérica para ahorrar espacio, y una estrategia de partición y escritura incremental para los niveles de mayor cardinalidad (a partir de aproximadamente 10 millones de filas), evitando construir una única tabla gigante en memoria.

Por razones de costo computacional, en la implementación actual del pipeline solo se entrenan de forma activa 4 de los 12 niveles: **total (L1)**, **dept (L4)**, **store (L6)** y **store_dept (L9)**. Los niveles a nivel item (L10-L12) —los de mayor volumen y mayor intermitencia de demanda— quedan fuera del alcance de entrenamiento activo de este trabajo y se retoman como línea futura (ver sección 10).

## 4. Análisis exploratorio

Antes de construir variables derivadas se realizó un análisis exploratorio de los datos, centrado en tres preguntas: cómo evoluciona la demanda en el tiempo, qué tan intermitente es, y qué series conviene tratar de forma diferenciada.

**Evolución temporal.** La venta diaria total pasa de 32.631 unidades el primer día de la serie (2011-01-29) a 54.338 unidades el último (2016-05-22), lo que evidencia una tendencia de crecimiento sostenido a lo largo de los casi 5 años y medio de historia, además de estacionalidad semanal y mensual visible en los desgloses por día de la semana, mes y tienda.

<!-- 📊 SUGERENCIA GRÁFICO: serie de tiempo de venta diaria total 2011-2016 (línea), y heatmaps tienda × día-de-semana y departamento × mes, generados de forma inline en notebooks/01_eda.ipynb (secciones "By date", "Store-DOW", "Dept-Month") — no están guardados como PNG hoy; exportarlos con plt.savefig() antes de insertarlos en el Word. -->

**Clasificación de la demanda (Syntetos-Boylan-Croston).** Dado que este es un problema de demanda de retail con muchas combinaciones producto-tienda de venta baja o esporádica, se calculó para cada serie el ADI (*Average Demand Interval*, espaciado promedio entre días con venta) y el CV² (coeficiente de variación al cuadrado de la demanda positiva), y se clasificó cada serie en una de 4 categorías siguiendo el criterio de Syntetos, Boylan y Croston (2005): **Smooth**, **Erratic**, **Intermittent** y **Lumpy**, usando los cortes estándar de la literatura (ADI = 1,32; CV² = 0,49).

<!-- 📊 SUGERENCIA GRÁFICO: gráfico de torta o barras con el % de series en cada categoría SBC (Smooth/Erratic/Intermittent/Lumpy) — calculado en notebooks/01_eda.py, función add_sbc_class(). -->

**Intermitencia.** A nivel fila serie-día, el **67,998% de las observaciones tiene venta 0**. Esto no es ruido: es la característica estructural más importante del dataset y condiciona buena parte de las decisiones metodológicas posteriores — el uso de WAPE/WRMSSE en vez de métricas porcentuales clásicas (sección 6), la construcción de variables específicas de intermitencia (sección 5) y la elección del objetivo de entrenamiento Tweedie para los niveles más finos (sección 7).

<!-- 🔢 SUGERENCIA DATO: destacar el 67,998% de filas con venta 0 como cifra/callout visual (por ejemplo un recuadro o KPI grande) — es el dato que más justifica el resto de las decisiones de diseño del trabajo. -->

**Muestras representativas.** Para la exploración manual se seleccionó una muestra estratificada de 300 series (100 de mayor, mediana y menor facturación respectivamente) y se graficaron ventanas de 90 días de series individuales para inspeccionar visualmente patrones de estacionalidad, promociones y quiebres de stock.

<!-- 📊 SUGERENCIA GRÁFICO: 2-3 gráficos de línea de series individuales (una "Smooth" y una "Lumpy", por contraste) desde la sección "Samples" de notebooks/01_eda.py — sirven para ilustrar visualmente por qué una talla única de modelo no basta. -->

## 5. Feature engineering

A partir del dataset base se construyó un conjunto de **más de 190 variables candidatas**, organizadas en las siguientes familias:

**Lags y ventanas móviles.** Para la granularidad diaria se generan lags en 1-28 (día a día) más lags "gruesos" en 35, 42, 56, 91, 182 y 364 días, pensados para capturar ciclos mensuales, trimestrales y anuales. Sobre varios de esos lags "ancla" (1, 7, 28, 91 y 364) se calculan además medias, desvíos, mínimos y máximos móviles, variables de *momentum* (diferencia entre ventanas cortas y largas) y medias estacionales, todo calculado de forma vectorizada sobre el historial completo de cada serie.

**Variables de calendario.** Componentes cíclicos seno/coseno del día de la semana, del mes y del día del año (para evitar la discontinuidad artificial de tratar, por ejemplo, diciembre y enero como extremos opuestos), además de trimestre e indicadores de inicio/fin de mes.

**Variables de precio.** Precio relativo al máximo histórico de la serie, volatilidad del precio en ventana móvil de 90 días, y días transcurridos desde el último cambio de precio.

**Eventos y calendario comercial.** Días hasta/desde el evento más cercano (genérico), más un tratamiento diferenciado para los dos eventos de mayor impacto empírico detectado en los datos: **Thanksgiving** (caída promedio de −39% en ventas) y **Año Nuevo** (−25%), ambos verificados sobre una muestra de 350 observaciones. También se incluyen indicadores de cierre de tienda (Navidad y ventas anómalamente bajas) y días hasta el próximo cierre conocido.

<!-- 🔢 SUGERENCIA DATO: tabla resumen "familia de variable → nº aproximado de columnas" (lags/rolling, calendario, precio, eventos, encoding, intermitencia) ayudaría a visualizar de un vistazo cómo se compone el total de ~190 variables. -->

**Target encoding jerárquico.** Media y desvío móvil (ventana de 28 días) de la venta diaria promedio por `item_id`, `dept_id` y `cat_id`, con *shrinkage* bayesiano hacia el nivel padre (peso del prior = 10) para evitar sobreajuste en combinaciones con poco historial, más medias condicionales expandidas por fin de semana, SNAP y presencia de evento.

**Variables de intermitencia.** Racha de días consecutivos en cero, porcentaje de días en cero en ventanas de 28 y 90 días, ADI expandido, CV² móvil a 90 días, e indicador de posible quiebre de stock (racha de al menos 7 días en cero estando el producto activo y con precio vigente).

Todas las variables se construyen íntegramente en Polars (lectura diferida, cálculo vectorizado) y solo se convierten a un formato tabular tradicional al final del proceso —paso requerido por la librería de forecasting utilizada—, aplicando en ese momento la misma optimización de tipos de datos descrita en la sección 3.

<!-- ⚠️ REVISAR ANTES DE ENTREGAR: para un forecast realmente hacia adelante (no backtesting), varias de estas variables (p. ej. las de intermitencia y algunas de encoding) asumen que se conoce el pasado inmediato de la serie; el propio código deja anotado que haría falta "congelar" esos valores en su último dato conocido para producción — ver sección 9. -->

## 6. Experimentación y evaluación comparativa

**Métricas utilizadas.** El dataset M5 combina dos particularidades que ninguna métrica por sí sola captura bien: demanda intermitente (las métricas de error porcentual como MAPE se indefinen o se disparan cuando la venta real es 0) y una jerarquía de escalas muy amplia (entre el nivel total y el nivel item-tienda hay varios órdenes de magnitud de diferencia en volumen, por lo que un error absoluto crudo no es comparable entre niveles). Por eso se reportan de forma conjunta:

- **Métricas puntuales** (por fila, agregadas sobre todo el horizonte evaluado): **WAPE** (error absoluto total como fracción de la venta total — la métrica de lectura rápida principal de este trabajo), **Bias** (mismo denominador que WAPE, indica sobre o subestimación), **MAE**, **RMSE**, **SMAPE** (secundaria, se infla con demanda intermitente) y **RMSLE**.
- **Métricas escaladas** (por serie, normalizadas contra un benchmark *naive* calculado *in-sample* sobre el propio train de esa serie, y agregadas ponderando por venta en $): **RMSSE** y su versión ponderada **WRMSSE** —la métrica oficial de ranking de la competencia M5—, y **MASE/WMASE** (Hyndman y Koehler, 2006), más robusta a *outliers* puntuales.
- **Tracking Signal** (Brown, 1959), para detectar sesgo sistemático sostenido en el tiempo (no solo su magnitud puntual).
- **SPEC** (*Stock-keeping-oriented Prediction Error Costs*; Martin, Hewamalage y Bergmeir), que a diferencia de las anteriores no solo mide la magnitud del error sino *cuándo* ocurre, con costo asimétrico entre quedarse corto (quiebre de stock, ponderado 0,75) y quedarse largo (sobre-stock, ponderado 0,25) — la métrica más alineada con el objetivo de negocio de reposición de inventario, aunque no se usa como objetivo de entrenamiento por no ser directamente diferenciable.

<!-- 🔢 SUGERENCIA DATO: docs/metricas_error.md (dentro del propio repo) contiene ejemplos numéricos sintéticos paso a paso de cada métrica, verificados contra el código real — es material directamente reutilizable como anexo de metodología. -->

**Comparación de familias de modelos.** Sobre el nivel total (L1) se comparó de forma sistemática el desempeño en validación de 21 configuraciones, agrupadas en tres familias: modelos *naive* (última observación, medias móviles, deriva, naive estacional a 7 y 365 días), modelos estadísticos clásicos (SARIMA, ETS/Holt-Winters, Theta, TBATS, Prophet) y modelos de Machine Learning (LightGBM, XGBoost, CatBoost, HistGradientBoosting, Ridge). Los resultados más relevantes (ordenados por WRMSSE) fueron:

| Modelo | Familia | WAPE | WRMSSE |
|---|---|---|---|
| HistGradientBoosting | ML | 5,89% | 0,541 |
| **LightGBM** | ML | 5,85% | 0,545 |
| CatBoost | ML | 6,02% | 0,567 |
| Ridge | ML | 6,16% | 0,590 |
| XGBoost | ML | 6,65% | 0,618 |
| ETS (Holt-Winters) | Estadístico | 9,53% | 0,828 |
| Theta | Estadístico | 10,20% | 0,880 |
| Prophet | Estadístico | 11,12% | 0,925 |
| Naive estacional (7d) | Naive | 12,06% | 0,987 |
| SARIMA | Estadístico | 13,14% | 1,070 |
| TBATS | Estadístico | 14,76% | 1,189 |
| Naive (último valor) | Naive | 16,89% | 1,317 |

<!-- 📊 SUGERENCIA GRÁFICO: barras horizontal de WRMSSE por modelo (los 21, o al menos los 12 de la tabla) — los datos ya están calculados en artifacts/results/level_01_daily_total_sales_model_comparison.csv, solo falta graficarlos. -->

Los dos mejores modelos (HistGradientBoosting y LightGBM) quedan prácticamente empatados y muy por delante del resto de familias: la brecha con el mejor modelo estadístico (ETS) es de más de 3,5 puntos de WAPE, y con el mejor *naive*, de más de 6 puntos. Se optó por **LightGBM** como modelo a optimizar en la siguiente fase, no solo por su desempeño competitivo sino por razones prácticas de ecosistema: soporte nativo para el método de explicabilidad exacta de SHAP en modelos de árboles (sección 8) y una integración madura con Optuna, que incluye un mecanismo de poda temprana de búsquedas poco prometedoras específico para LightGBM (sección 7).

<!-- 🔢 SUGERENCIA DATO: la tabla completa de 21 modelos (incluye también drift, historical mean y medias móviles de 7/14/21/28/35 días) está en artifacts/results/level_01_daily_total_sales_model_comparison.csv por si se prefiere anexarla completa. -->

## 7. Optimización de los modelos seleccionados

**Selección de variables (eliminación hacia atrás + importancia por permutación).** Sobre el conjunto de más de 190 variables candidatas (sección 5) se aplicó un proceso de eliminación hacia atrás: en cada paso se calcula la importancia por permutación de cada variable activa (sobre una muestra de validación de 15.000 filas y 5 repeticiones), se ordenan de menor a mayor importancia, y se evalúa eliminar la menos importante; la eliminación se acepta solo si el error de validación (RMSE o la métrica de Tweedie, según el nivel) no empeora, repitiendo el proceso hasta que una pasada completa no elimina ninguna variable. El resultado es un conjunto final de variables específico por nivel:

| Nivel | Variables finales |
|---|---|
| L1 (total) | 144 |
| L4 (dept) | 140 |
| L6 (store) | 161 |

**Optimización de hiperparámetros (Optuna).** Sobre el modelo con las variables ya seleccionadas se ejecuta una búsqueda bayesiana de hiperparámetros con Optuna, usando el algoritmo Tree-structured Parzen Estimator (TPE, con semilla fija para reproducibilidad) y un mecanismo de poda por mediana que corta pruebas poco prometedoras tempranamente. El espacio de búsqueda cubre:

| Hiperparámetro | Qué controla | Rango explorado |
|---|---|---|
| Tasa de aprendizaje | velocidad de ajuste del modelo en cada iteración | 0,03 – 0,15 |
| Nº de hojas por árbol | complejidad de cada árbol individual | 31 – 255 |
| Profundidad máxima | complejidad de cada árbol individual | 5 – 10 |
| Mínimo de muestras por hoja | evita divisiones sobre muy pocos datos | 20 – 200 |
| Submuestreo de filas | fracción de datos usada en cada árbol | 0,6 – 1,0 |
| Submuestreo de columnas | fracción de variables usada en cada árbol | 0,6 – 1,0 |
| Regularización (L1 / L2) | penaliza la complejidad para evitar sobreajuste | 0,001 – 5 |

El número máximo de árboles se fija en 1.500 con parada temprana (paciencia de 30 iteraciones), y cada estudio de optimización se guarda en una base de datos local para poder retomarlo entre ejecuciones. El presupuesto de búsqueda se ajustó recientemente de 100.000 pruebas/15 minutos a **500 pruebas/10 minutos**: en la práctica, el límite real siempre fue el tiempo y no la cantidad de pruebas (el estudio del nivel tienda-departamento completó apenas 35 pruebas en los 15 minutos disponibles), por lo que reducir el número máximo no cambia el resultado práctico y simplifica la configuración.

<!-- 📊 SUGERENCIA GRÁFICO: historial de convergencia de Optuna (optuna.visualization.plot_optimization_history) leyendo directamente artifacts/optuna_study.db — muestra visualmente cómo se estabiliza el error a medida que avanzan los trials. -->

Los hiperparámetros finales seleccionados por nivel fueron:

| Nivel | Tasa de aprendizaje | Nº hojas | Profundidad máx. | Mín. muestras/hoja | Submuestreo filas | Submuestreo columnas | Árboles finales |
|---|---|---|---|---|---|---|---|
| L1 (total) | 0,150 | 61 | 5 | 36 | 0,967 | 0,803 | 352 |
| L4 (dept) | 0,142 | 229 | 9 | 26 | 0,986 | 0,870 | 92 |
| L6 (store) | 0,129 | 89 | 7 | 121 | 0,952 | 0,869 | 92 |

**Objetivo de entrenamiento.** Para los niveles 1 a 10 se entrena minimizando el error cuadrático medio (RMSE). Para los niveles 11 y 12 (item-tienda, donde la mayoría de los días tienen venta 0) el pipeline ya contempla un objetivo de tipo **Tweedie**: a diferencia de RMSE, esta función penaliza el error en términos relativos a la magnitud esperada, evitando que el modelo aprenda a "regalarse" los días en cero por ser la opción más barata en términos de error cuadrático. Este objetivo ya está implementado y listo para usarse, aunque los niveles 11-12 no forman parte del alcance de entrenamiento activo de este trabajo (ver sección 10).

**Resultados finales en test.** Tras la selección de variables y la optimización de hiperparámetros, el desempeño del modelo final sobre el conjunto de test (28 días fuera de muestra, posteriores a validación) fue:

| Nivel | WAPE | WRMSSE | MAE | RMSE | Bias | MASE | Nº series |
|---|---|---|---|---|---|---|---|
| L1 (total) | 7,47% | 0,707 | 3.287,5 | 4.181,3 | −7,42% | 0,773 | 1 |
| L4 (dept) | 6,89% | 0,584 | 432,8 | 684,5 | −4,30% | 0,638 | 7 |
| L6 (store) | 6,29% | 0,530 | 276,8 | 376,7 | −2,51% | 0,559 | 10 |

*(Nota: estos valores de test no son directamente comparables con la tabla de validación de la sección 6, ya que corresponden a una ventana temporal distinta y posterior — no debe interpretarse como que el modelo "empeoró" tras la optimización.)*

Adicionalmente, se entrenó un modelo para el nivel **tienda-departamento (L9, 70 series)**, aún en vías de consolidación dentro de la suite estándar de test, con resultados preliminares de WAPE 8,68% y WRMSSE 0,616 en test (tasa de aprendizaje de 0,033, 244 hojas por árbol, profundidad máxima de 10 y 1.360 árboles finales — un modelo notablemente más complejo que los otros tres, coherente con tratarse de 70 series heterogéneas entrenadas conjuntamente).

Un hallazgo interesante es que los niveles **dept (L4)** y **store (L6)** generalizan mejor en test que el nivel **total (L1)**, pese a ser (levemente) más granulares: agregar todas las categorías y tiendas en una sola serie parece diluir patrones específicos de categoría o tienda que sí son explotables cuando se modelan por separado.

<!-- 🔢 SUGERENCIA DATO: esta tabla de test es el resultado más citable de todo el trabajo — buena candidata a aparecer también en el resumen ejecutivo o en la introducción. -->

## 8. Interpretabilidad

Sobre el modelo final de cada nivel se calculan valores SHAP (*SHapley Additive exPlanations*), un método exacto y eficiente para explicar modelos de árboles como LightGBM, sobre una muestra de hasta 300.000 filas. Para el nivel total (L1) se generaron los siguientes gráficos:

![SHAP – importancia media absoluta (L1, total)](../artifacts/plots/level_01_daily_total_sales_shap_bar.png)

<!-- 🖼️ IMAGEN YA INSERTADA: artifacts/plots/level_01_daily_total_sales_shap_bar.png (ranking de importancia media |SHAP|). -->

![SHAP – beeswarm (L1, total)](../artifacts/plots/level_01_daily_total_sales_shap_beeswarm.png)

<!-- 🖼️ IMAGEN YA INSERTADA: artifacts/plots/level_01_daily_total_sales_shap_beeswarm.png (distribución de impacto por variable y por observación). -->

![SHAP – dependencia del lag de 28 días (L1, total)](../artifacts/plots/level_01_daily_total_sales_shap_dependence_lag28.png)

<!-- 🖼️ IMAGEN YA INSERTADA: artifacts/plots/level_01_daily_total_sales_shap_dependence_lag28.png (relación entre la venta de hace 28 días y su efecto en la predicción). -->

Para el nivel total, la variable con mayor impacto es, con claridad, **el lag de 28 días** (venta de hace 4 semanas) — coherente con un ciclo de compra mensual/de reposición. El resto de las variables relevantes son lags igualmente largos (7, 91 y 182 días) y el indicador SNAP.

A nivel de mayor granularidad (tienda-departamento, L9) el orden cambia: dominan los lags cortos (el de 1 día, con importancia media 72,4, por delante del de 28 días con 52,4 y el de 7 días con 44,9), seguidos de una media móvil corta a 7 días (33,3) y el departamento del producto (25,9). La lectura es consistente con la intuición de negocio: cuanto más fina la granularidad, más pesa la inercia de corto plazo (qué se vendió ayer) frente al patrón cíclico de reposición que domina cuando se mira la demanda ya agregada a nivel total.

<!-- 🖼️ SUGERENCIA IMAGEN: regenerar el mismo trío de gráficos SHAP (bar/beeswarm/dependence) para L4, L6 y L9 — hoy solo existen en disco para L1 (artifacts/plots/); ejecutar de nuevo scripts/train_dataset.py con run_shap=True para esos niveles. -->

**Explicación a nivel de instancia.** Además de la interpretabilidad global, el proyecto incluye una función de explicación puntual que genera un gráfico de tipo *waterfall* (cascada) de SHAP para una combinación serie-fecha específica, mostrando qué variables empujaron la predicción hacia arriba o hacia abajo respecto del valor base. Se utiliza para diagnosticar en detalle los días de mejor y peor error dentro de una serie elegida.

<!-- ⚠️ REVISAR ANTES DE ENTREGAR: en notebooks/03_predictions.ipynb, la celda que fija la serie a analizar en detalle (variable SERIES) está hardcodeada con un id de nivel dept, que no existe en el artifact de nivel store actualmente cargado en el notebook — da como resultado una tabla vacía. Actualizar el id de serie antes de re-ejecutar esa sección para generar las imágenes de detalle por serie. -->

## 9. Productivización vía API

> **Nota:** esta sección describe una **propuesta de arquitectura**, no una implementación ya construida. A la fecha de este informe, el repositorio no contiene código de servicio (API, contenedores, etc.) — el resultado de este trabajo son los artefactos de modelo entrenados y los procesos que los generan y evalúan por línea de comandos. La productivización queda como línea de trabajo inmediato posterior a esta memoria.

El diseño actual de los artefactos ya facilita este paso: cada modelo se exporta como un único objeto serializado que empaqueta el modelo LightGBM entrenado junto con sus metadatos (variables usadas, variables categóricas, columnas de identificación, fecha de corte de entrenamiento, métricas de validación/test), sin datos crudos — es decir, ya sigue un patrón de "artefacto liviano" razonable para servir en producción.

**Arquitectura propuesta:**

- **Framework:** FastAPI, por su soporte nativo de validación de esquemas y documentación automática de la API.
- **Endpoint principal:** una ruta de predicción que reciba el nivel de agregación, el identificador de serie (item, tienda, departamento, etc. según el nivel) y el horizonte de predicción solicitado; devolviendo la predicción puntual junto con metadatos de contexto (fecha de corte del modelo, métricas de validación del nivel usado).
- **Carga de modelo:** al iniciar el servicio, cargar en memoria los artefactos de modelo relevantes (patrón ya usado hoy en los procesos de evaluación por línea de comandos), evitando releerlos en cada solicitud.
- **Reconstrucción de features:** el mayor desafío de llevar esto a producción no es servir el modelo sino las variables de entrada — hoy la reconstrucción de variables asume que se dispone de todo el historial hasta la fecha objetivo. Para un forecast genuinamente hacia adelante (no *backtesting* sobre fechas ya conocidas) haría falta un paso adicional que "congele" las variables cuasi-estáticas (intermitencia, codificaciones históricas) en su último valor conocido, tal como ya se identificó durante el desarrollo del proyecto.
- **Empaquetado y despliegue:** contenedorización con Docker (hoy inexistente en el repo) y, si se prevé reentrenamiento periódico, un registro de modelos/versionado. El propio documento de propuesta original del TFM contemplaba usar **MLflow** para *tracking* de experimentos, herramienta que finalmente no se llegó a integrar y que sería natural incorporar en este punto, tanto para *tracking* de entrenamiento como para registro/versionado de los modelos servidos por la API.
- **Pruebas:** el repositorio no cuenta hoy con una suite de tests automatizados (solo una verificación manual puntual de la lógica de *split* temporal); antes de exponer un servicio de cara a sistemas productivos convendría cubrir al menos la carga de artefactos, el contrato de entrada/salida de la API y la reconstrucción de features con tests automatizados.

<!-- 📊 SUGERENCIA GRÁFICO: un diagrama simple de arquitectura (cliente → API FastAPI → carga de artifact .pkl → reconstrucción de features → predicción) ayudaría mucho aquí a comunicar la propuesta visualmente. -->

## 10. Conclusiones y líneas futuras

Este trabajo construyó un pipeline completo de previsión de demanda sobre el dataset M5 (Walmart), cubriendo los 12 niveles oficiales de agregación de la competencia, con una implementación especialmente cuidada en el uso de memoria (DuckDB + Polars) para poder operar sobre un dataset de hasta 59 millones de filas sin recurrir a infraestructura distribuida. Sobre 4 de esos niveles (total, departamento, tienda y tienda-departamento) se ejecutó el ciclo completo: comparación sistemática de 21 configuraciones de modelado (*naive*, estadísticas clásicas y de Machine Learning), selección de variables por eliminación *backward* guiada por *permutation importance*, optimización bayesiana de hiperparámetros con Optuna, e interpretabilidad con SHAP.

**Principales hallazgos:**

- **LightGBM** iguala o supera a alternativas más complejas (HistGradientBoosting, XGBoost, CatBoost) y a modelos estadísticos clásicos (SARIMA, ETS, Theta, Prophet, TBATS), con una ventaja de más de 6 puntos de WAPE sobre el mejor *naive* — confirmando que, para este problema, la ganancia de usar Machine Learning sobre líneas base simples es sustancial.
- El desempeño final en test (WAPE entre 6,3% y 7,5%, WRMSSE entre 0,53 y 0,71 según nivel) es consistente y estable entre niveles, con **dept** y **store** generalizando levemente mejor que **total**, lo que sugiere que agregar en exceso diluye patrones específicos de categoría o tienda.
- La importancia relativa de las variables cambia con la granularidad: a nivel total domina el ciclo mensual (venta de hace 28 días), mientras que a nivel más fino (tienda-departamento) domina la inercia de corto plazo (venta de hace 1 y 7 días).
- El **67,998%** de intermitencia a nivel serie-día es la característica estructural que más condicionó las decisiones metodológicas del trabajo (elección de métricas, features de intermitencia, objetivo Tweedie).

**Limitaciones y líneas futuras:**

- **Niveles item-level (L10-L12).** Concentran el mayor volumen y la mayor intermitencia del dataset, y quedaron fuera del alcance de entrenamiento activo por costo computacional. El objetivo Tweedie ya está implementado para ellos; la línea futura natural es extender el pipeline (probablemente con procesamiento distribuido o entrenamiento por segmentos más agresivo) para cubrirlos.
- **Productivización vía API.** Como se detalla en la sección 9, queda como diseño propuesto, no implementado — es la línea de trabajo inmediato más directa a partir de esta memoria.
- **Seguimiento de experimentos.** La propuesta original de este TFM contemplaba usar MLflow; en la práctica el *tracking* se resolvió de forma más artesanal (SQLite de Optuna + artefactos CSV/JSON/pickle). Integrar MLflow (u otra herramienta equivalente) mejoraría la trazabilidad y reproducibilidad a medida que crezca el número de niveles/experimentos.
- **Pruebas automatizadas.** No existe hoy una suite de tests (solo una verificación manual de la lógica de partición temporal); es una brecha a cerrar antes de cualquier despliegue productivo.
- **Backtesting vs. forecasting real.** La evaluación actual predice sobre fechas históricas ya conocidas (con todo su historial disponible). Un despliegue productivo real requiere el paso adicional de "congelar" variables cuasi-estáticas para fechas verdaderamente futuras, ya anotado como pendiente en el propio código.
- **Modelos probabilísticos.** Todo el trabajo se centra en predicción puntual; para decisiones de *stock* de seguridad e inventario, una línea natural es extender a predicción por cuantiles o distribuciones completas de demanda.
- **Nivel store_dept (L9).** Los resultados presentados en la sección 7 son preliminares (artefacto de una iteración anterior del pipeline); consolidarlo dentro de la suite estándar de entrenamiento/test es un paso pendiente de corto plazo.
- **Horizontes de predicción acumulados.** El pipeline ya soporta entrenar, además de la venta diaria/semanal puntual, objetivos de demanda acumulada a distintos horizontes (p. ej. la suma de venta esperada en los próximos 7, 14 o 28 días), útiles para decisiones de reposición a distintos plazos. Por acotar el alcance de esta entrega se priorizó consolidar bien el objetivo de venta puntual en los 4 niveles activos antes que extender a estos horizontes adicionales; queda como línea de trabajo inmediata siguiente.
- **Presupuesto de ajuste.** Los resultados de la sección 7 surgen de una primera pasada completa del pipeline (bench, selección de variables, Optuna e interpretabilidad) sobre los 4 niveles activos; con más tiempo disponible, una ronda adicional de ajuste fino de hiperparámetros y de revisión de la selección de variables por nivel es razonable esperar que mejore levemente estos números.

<!-- 🔢 SUGERENCIA DATO: si se dispone de tiempo, un cuadro resumen final tipo "objetivo propuesto vs. logrado" (comparando el índice de la propuesta de junio, docs/propuesta.md, contra lo efectivamente entregado) sería un cierre muy efectivo para la defensa. -->

## 11. Bibliografía

- Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). *On the categorization of demand patterns*. Journal of the Operational Research Society, 56(5), 495–503.
- Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill.
- Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679–688.
- Martin, D., Hewamalage, H., & Bergmeir, C. (2022). *Similarity-based cost forecasting error metrics for the stock-keeping context* (SPEC). arXiv:2004.10537.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). *The M5 Accuracy competition: Results, findings, and conclusions*. International Journal of Forecasting, 38(4), 1346–1364.
- Kaggle. *M5 Forecasting – Accuracy* [conjunto de datos y competencia]. https://www.kaggle.com/competitions/m5-forecasting-accuracy

<!-- 🔢 SUGERENCIA DATO: verificar el formato de citación exigido por la normativa de TFM del máster (APA/IEEE/etc.) y ajustar el listado anterior en consecuencia. -->
