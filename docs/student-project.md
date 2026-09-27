# Guía del proyecto estudiantil

## Primera etapa: datos estáticos

1. instala el SDK y descarga el corte inicial;
2. valida continuidad, duplicados, tipos y cobertura por estación;
3. realiza análisis exploratorio temporal y geográfico;
4. construye al menos dos baselines;
5. usa backtesting temporal y conserva evidencia de cada experimento;
6. define cómo versionarás modelo, features y cutoff.

## Segunda etapa: operación incremental

Cuando se active el reloj, GitHub Actions deberá:

1. consultar únicamente observaciones nuevas;
2. persistir el cursor o último timestamp procesado;
3. calcular métricas y señales de drift;
4. decidir si conserva o reentrena el modelo;
5. generar los cuatro horizontes solicitados;
6. enviar la predicción con versión y commit;
7. registrar éxito o error de la ejecución.

## Entregables mínimos

- repositorio reproducible;
- README con arquitectura y decisiones;
- pipeline automático en GitHub Actions;
- validación temporal y comparación contra baselines;
- monitoreo de datos y desempeño;
- estrategia explícita de reentrenamiento;
- historial de predicciones y modelos.

## Dashboard de monitoreo

El repositorio incluye un dashboard MLOps en `app/`, desplegable en Vercel. Su
objetivo es facilitar la decisión operativa sin reemplazar el pipeline:

- modelo activo, versiones anteriores y referencias de MLflow cuando existen;
- WAPE actual, baseline, ratio y tendencia histórica;
- PSI por variable, umbral, variables afectadas y máximo observado;
- estado `drift_alert`, `retrain` y explicación de la recomendación;
- historial de ejecuciones exitosas y fallidas de `model_drift_execution`;
- estado del pipeline, alertas y actualización periódica de datos.

La interfaz consulta `model`, `training_run`, `model_drift_check`,
`model_drift_execution`, `pipeline_execution` y `pipeline_sync_status`. No crea
tablas, no recalcula la lógica de decisión y no inicia ejecuciones de drift.

Consulta [dashboard.md](dashboard.md) para el despliegue. Nunca expongas claves
privadas de Supabase en el navegador: `SUPABASE_SERVICE_ROLE_KEY` solo debe
existir como variable server-side en Vercel.
