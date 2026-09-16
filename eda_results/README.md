# Resultados del EDA

Resultados tabulares del analisis exploratorio ejecutado el 2026-09-16.

Fuente: API publica `https://pulso-transmi.72-60-245-2.sslip.io`.

El analisis consulto unicamente endpoints GET y proceso las respuestas en memoria. Estas tablas son un snapshot del corte servido por la API; no son un dataset de entrenamiento.

Archivos:

- `dataset_summary.csv`: tamano, periodo y memoria aproximada por recurso.
- `schema.csv`: variables y tipos observados.
- `quality_summary.csv`: nulos, unicos y duplicados.
- `numeric_summary.csv`: estadistica descriptiva y outliers IQR.
- `categorical_summary.csv`: frecuencias de variables categoricas.
- `target_by_station.csv`: distribucion de `demand` por estacion.
- `correlations.csv`: correlaciones exploratorias.
- `temporal_coverage.csv`: cobertura por estacion.
- `special_values.csv`: valores especiales detectados.
- `risks_and_recommendations.csv`: riesgos y acciones sugeridas.

No se aplicaron limpieza, imputacion, transformacion, encoding, normalizacion ni entrenamiento.
