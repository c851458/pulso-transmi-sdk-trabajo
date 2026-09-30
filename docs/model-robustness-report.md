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

## Modelo robusto al drift (2026-09-30)

Se añadió el candidato `drift_robust_ensemble` (`src/robust_model.py`),
pensado para adaptarse a cambios fuertes en la distribución de los datos:

- **Pesos por recencia** (`RecencyWeighted`): cada fila pesa
  `0.5 ^ (antigüedad / RECENCY_HALF_LIFE_DAYS)`, con 14 días por defecto y un
  mínimo de 0,05. Tras un cambio de régimen, los datos nuevos dominan el ajuste
  sin descartar la historia.
- **Ensamble adaptativo** (`DriftRobustEnsemble`) de tres miembros diversos:
  - `hgb_poisson`: gradient boosting con pérdida Poisson, adecuada para conteos
    no negativos y cambios multiplicativos;
  - `hgb_absolute`: gradient boosting con error absoluto, que no se deja
    arrastrar por picos de demanda;
  - `ridge`: lineal, el único miembro que extrapola cuando una variable sale
    del rango de entrenamiento (los árboles se quedan planos).

  Cada miembro pesa `1 / MAE²` según su error en el último 15 % cronológico
  del entrenamiento; si un drift hace fallar a un miembro, pierde peso en el
  siguiente reentrenamiento.
- **Salvaguarda**: predicciones recortadas a `[0, 1,5 × demanda máxima vista]`,
  para que una extrapolación extrema no publique valores absurdos.
- **Selección consciente del drift**: los candidatos se ordenan por
  `cv_recent_mae` (MAE medio de los 2 folds temporales más recientes, los más
  parecidos a producción) y luego por estabilidad, en lugar del MAE promedio de
  todos los folds. También se registra `cv_worst_mae` (peor fold).

La clase vive en `src/` y no en el script de entrenamiento porque joblib guarda
referencias por módulo: el pipeline debe poder importarla al cargar el
artefacto.

### Comparación con los datos actuales (51.840 observaciones)

| Modelo | MAE CV medio | MAE folds recientes | MAE peor fold | MAE test | RMSE test | WAPE test | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **drift_robust_ensemble** | **49,27** | **45,39** | **59,97** | **46,35** | 76,24 | **12,7 %** | 0,945 |
| hist_gradient_boosting | 51,14 | 47,04 | 64,60 | 47,67 | **75,03** | 13,0 % | 0,946 |
| random_forest | 53,66 | 49,30 | 67,09 | 48,95 | 78,48 | 13,4 % | 0,941 |
| robust_ridge | 202,09 | 202,08 | 204,77 | 207,55 | 277,62 | 56,8 % | 0,267 |
| baseline_linear | 202,25 | 202,22 | 205,00 | 207,52 | 277,55 | 56,8 % | 0,267 |

- El ensamble reduce el MAE en los folds recientes un 3,5 % y en el peor fold
  un 7,2 % frente a `hist_gradient_boosting`; su RMSE es ligeramente mayor
  (+1,6 %), porque el miembro de error absoluto sacrifica algo en los picos.
- Pesos aprendidos: `hgb_poisson` 0,495, `hgb_absolute` 0,479, `ridge` 0,027.
  Con los datos actuales el Ridge casi no participa (MAE 201 en el tramo
  reciente); su peso solo crece si un drift deja a los árboles peor que él.
- Prueba de estrés: multiplicar `event_intensity` (rango de entrenamiento
  0-1) por 1,5, 2 y 3 mueve la predicción media de 507 a 510, 512 y 517: sin
  saltos ni valores negativos.
- Costo: unos 97 s de entrenamiento del candidato (2 min 45 s el script
  completo en local), dentro del límite de 30 min del workflow.

En producción: activo desde 2026-09-30 21:01 UTC como modelo #286
(`drift_robust_ensemble-303ab30ff133`), con MAE test 46,46 y accuracy
87,29 % en el entrenamiento del workflow.

`tests/test_robust_model.py` verifica los pesos por recencia, el recorte de
predicciones extremas, que la ponderación por recencia se adapta mejor a un
cambio de régimen simulado y que el modelo sobrevive al pickling.

## Limitaciones

El corte disponible contiene 45 dias y solo 12 estaciones. No hay todavia una
validacion fuera de muestra de varias semanas futuras ni variables lag de
 demanda. El modelo no debe considerarse estable frente a un cambio de regimen
hasta acumular nuevos ciclos y observar las metricas reales de produccion.
