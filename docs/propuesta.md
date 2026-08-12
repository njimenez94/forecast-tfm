Acá tienes la propuesta completa tal cual la entregaste el 18 de junio:

**1) Opción seleccionada**
```
1) Análisis de un dataset (orientación Data Scientist)
```


**2) Índice**

1. Introducción y contexto de negocio
2. Descripción del dataset M5
3. Diseño experimental
4. Análisis exploratorio (EDA)
5. Feature engineering
6. Experimentación y evaluación comparativa
7. Optimización de los modelos seleccionados
8. Interpretabilidad
9. Productivización vía API
10. Conclusiones y líneas futuras
Bibliografía
Anexos


**3) Descripción del TFM propuesto**

El trabajo aborda la previsión de demanda en un entorno de retail a
partir del dataset público M5 Forecasting (competición de Kaggle con
datos reales de ventas de Walmart), que recoge las ventas diarias a
nivel item-tienda durante casi cinco años, junto con información de
calendario, eventos y precios. El objetivo de negocio es anticipar la
demanda para apoyar decisiones de planificación de inventario,
reposición de stock y pricing.

El trabajo se estructura en dos fases. En una primera fase de
experimentación se evalúa de forma sistemática el desempeño de distintas
formulaciones del problema: diferentes definiciones del objetivo (venta
diaria, semanal y acumulada), distintos niveles de agregación, varios
horizontes de predicción y diferentes esquemas de entrenamiento. El
propósito es abarcar el mayor número de escenarios posibles y comparar
su error de forma homogénea, empleando métricas adecuadas a la previsión
de demanda (RMSE, MAE, WAPE y bias).

A partir de esta comparación se seleccionan los modelos más adecuados
según un doble criterio de error y utilidad para el negocio. Sobre los
modelos elegidos se realiza una segunda fase de optimización: tuning de
hiperparámetros con Optuna, búsqueda de variables y esquemas de
validación temporal, con seguimiento de experimentos mediante MLflow, e
interpretabilidad con SHAP.

```
Nota: en el índice quedó "Productivización vía API" como punto 9 pero en la descripción se recortó (no se menciona la API explícitamente al final) — eso quedó pendiente de desarrollar en la memoria de septiembre.
```