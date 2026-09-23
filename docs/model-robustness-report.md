# Evaluacion de robustez del modelo

## Auditoria del baseline

El problema es una regresion de demanda (`demand`) por estacion y timestamp. El
baseline original usaba `LinearRegression` con one-hot para `station_id` y
`corridor`, variables meteorologicas/eventos y codificacion ciclica de hora y
dia de semana.

La particion ya era cronologica, lo cual evita mezclar futuro y pasado, y el
`TimeSeriesSplit` se aplicaba solo sobre el bloque de entrenamiento. No se
detecto una fuga temporal explicita en las variables existentes: no hay lags ni
variables calculadas usando el target futuro. Los riesgos principales eran:

- regresion sin regularizacion;
- ausencia de imputacion para nulos y valores infinitos;
- ausencia de escalado robusto para variables numericas;
- falta de manejo explicito de categorias desconocidas;
- un solo modelo y un solo criterio de seleccion;
- falta de dispersion de la metrica entre ventanas temporales.

## Cambios

El entrenamiento ahora:

1. valida joins, target no negativo y claves estacion/timestamp;
2. reemplaza infinitos por `NaN` y elimina claves duplicadas conservando el
   ultimo registro;
3. imputa numericas con mediana y categoricas con la moda dentro del pipeline;
4. usa `OneHotEncoder(handle_unknown="ignore")`;
5. usa `RobustScaler`, menos sensible a outliers que el escalado por media y
   desviacion;
6. compara modelos con `TimeSeriesSplit(n_splits=5)`;
7. reporta media y desviacion de MAE, RMSE y R2 entre ventanas;
8. guarda el pipeline completo para que entrenamiento e inferencia compartan
   exactamente el mismo preprocessing.

Los candidatos son el baseline lineal, Ridge con `alpha=10`, un
`RandomForestRegressor` acotado (`n_estimators=200`, `max_depth=18`,
`min_samples_leaf=5`) y
`HistGradientBoostingRegressor` con complejidad indirectamente limitada por
`max_leaf_nodes=31`, `min_samples_leaf=30`, `learning_rate=0.05`,
`l2_regularization=1.0` y early stopping.

## Comparacion reproducible

Resultados sobre el corte temporal actual de 51.840 observaciones. La columna
CV es la media de cinco ventanas temporales; `CV MAE std` mide estabilidad.

| Modelo | CV MAE | CV RMSE | CV R2 | CV MAE std | Holdout MAE | Holdout RMSE | WAPE | Tiempo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline lineal | 202.25 | 272.51 | 0.266 | 1.98 | 207.52 | 277.55 | 0.568 | 0.83 s |
| Ridge robusto | 202.09 | 272.42 | 0.266 | 2.04 | 207.55 | 277.62 | 0.568 | 0.58 s |
| Random Forest | 47.37 | 76.83 | 0.941 | 5.27 | 44.11 | 72.31 | 0.121 | 15.63 s |
| HistGradientBoosting | 51.14 | 79.12 | 0.937 | 6.80 | 47.67 | 75.03 | 0.130 | 5.96 s |

El modelo seleccionado es `random_forest`: mejora de forma consistente la MAE y
RMSE en validacion temporal, no solo en el holdout, y su dispersion relativa
entre ventanas (`std / mean MAE`) es aproximadamente 11.1%, menor que el 13.3%
de boosting. Su coste de entrenamiento es aproximadamente 2.6 veces mayor que
boosting, pero sigue siendo bajo para el reentrenamiento programado del
proyecto. Boosting queda como alternativa si el tiempo de ejecucion pasa a ser
la prioridad.

Los resultados completos quedan en `artifacts/baseline/metrics.json` y
`artifacts/baseline/model_comparison.csv`. El pipeline serializado queda en
`artifacts/baseline/model_and_metrics.joblib`.

## Produccion y monitoreo

La inferencia usa el pipeline serializado completo y mantiene las mismas
variables de entrada. Las categorias nuevas se ignoran de forma segura y los
nulos numericos se imputan con estadisticas aprendidas solo del entrenamiento.

En cada reentrenamiento conviene registrar:

- MAE, RMSE, WAPE y R2 por ventana temporal y por estacion;
- porcentaje de nulos, infinitos, duplicados y categorias nuevas;
- distribucion de `demand` y de las variables numericas frente al entrenamiento;
- error rolling de las ultimas 24 horas cuando llegue el valor real;
- alertas si la MAE supera el baseline o si el drift de una variable cambia de
  forma sostenida.

Se recomienda reentrenar cuando haya suficiente dato nuevo o cuando la MAE
rolling supere un umbral acordado durante dos ventanas consecutivas. El
boosting puede ser reemplazado por Ridge si el tiempo de ejecucion o la
variabilidad temporal se vuelven mas importantes que la mejora de error.

## Limitaciones

El corte disponible contiene 45 dias y solo 12 estaciones. No hay todavia una
validacion fuera de muestra de varias semanas futuras ni variables lag de
 demanda. El modelo no debe considerarse estable frente a un cambio de regimen
hasta acumular nuevos ciclos y observar las metricas reales de produccion.
