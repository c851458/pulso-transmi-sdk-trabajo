# Fiabilidad del pipeline de inferencia

## Problema auditado

Una ejecucion verde de GitHub Actions no demostraba que la API hubiera recibido
las predicciones: el proceso podia terminar despues de calcular artefactos sin
validar el cuerpo de respuesta, el identificador de submission o la cantidad
de predicciones confirmadas. Ademas, habia dos workflows capaces de publicar y
eso podia crear carreras entre ejecuciones.

## Controles implementados

- Validacion de metricas obligatorias: `mae`, `rmse`, `wape` y `accuracy`.
- Rechazo de `None`, vacios, `NaN`, `Infinity` y conteos inconsistentes.
- Validacion del payload completo antes del POST.
- `run_id` determinista mediante `cycle_id + model_version`.
- Outbox local en `artifacts/outbox/`; se elimina solo despues de confirmacion.
- Upload del outbox y del estado del pipeline como artefacto de GitHub cuando
  la ejecucion falla.
- Reintentos limitados para lecturas GET de Supabase ante timeout, red, 408,
  429 y 5xx, con backoff de `1s, 2s`; las escrituras no se reintentan
  automáticamente para evitar duplicados.
- Sin reintentos para 400, 401, 403, 404 y otros errores permanentes.
- La respuesta debe contener `submission_id`, `id` u `operation_id` (o una
  confirmacion booleana), estado valido y, cuando la API lo entrega, exactamente
  el numero esperado de predicciones y el ciclo correcto.
- El pipeline solo escribe `SUCCESS` despues de confirmar API; de lo contrario
  escribe `FAILED` y devuelve codigo distinto de cero.
- `concurrency` usa un grupo compartido para ingestión, drift, entrenamiento y
  publicación, evitando que varios workflows escriban en Supabase al mismo
  tiempo.
- Las tareas reutilizan un cliente REST por ejecución, con un pool HTTP pequeño
  (máximo cuatro conexiones y keep-alive). Solo las lecturas `GET` tienen
  reintentos acotados; las escrituras no se repiten automáticamente.
- `model_drift.yml` conserva su schedule de `*/10 * * * *` y
  `workflow_dispatch`; `pipeline.yml` ejecuta ingesta, entrenamiento y publicación
  cada diez minutos mediante `workflow_dispatch` disparado desde cron-job.org; no
  tiene `schedule` propio para no duplicar ejecuciones.
- `data_pipeline.yml` permanece manual porque la ingesta ya forma parte del pipeline
  automático y un segundo schedule produciría ejecuciones duplicadas.
- `supabase/migrations/20260927180000_monitoring_query_indexes.sql` añade
  índices no destructivos para las consultas frecuentes de monitoreo. Debe
  aplicarse desde Supabase cuando la Data API vuelva a estar disponible.
- `supabase/migrations/20260927190000_cpu_query_indexes.sql` añade índices
  compuestos para las consultas incrementales de ingestión y drift.
- `supabase/migrations/20260927200000_model_history_retention.sql` crea
  `prune_model_history`. Cada vez que se registra un modelo nuevo, el pipeline
  borra las predicciones de evaluación y `monitoring_metric` de los modelos
  anteriores, los marca `retired` y elimina su `artifact_base64` (el artefacto
  sigue en MLflow). Las predicciones enviadas a la API, las observaciones y las
  ejecuciones se conservan. Sin esta retención, el reentrenamiento periódico llenó
  el disco de 2 GB del plan free y Postgres dejó de responder (HTTP 503 PGRST002).
- `pipeline_sync_status` expone al portal el estado de sincronización y
  `last_updated_at`; el frontend puede marcar datos como stale según su propia
  ventana de frescura sin ejecutar la ingesta.

## Estado final esperado

```text
Datos correctos
  -> inferencia completada
  -> metricas validas
  -> payload valido
  -> API contactada
  -> respuesta confirmada
  -> SUCCESS
```

La API de competencia no ofrece un endpoint independiente para metricas. La
submission confirma predicciones y contrato; las metricas de evaluacion se
persisten en Supabase dentro de `training_run` y `monitoring_metric`. El job no
trata una respuesta HTTP exitosa sin confirmacion semantica como exito.

## Pruebas

La suite cubre respuesta aceptada, reintentos ante 503, no-reintento ante 401 y
rechazo de respuesta sin confirmacion. Tambien se validaron manualmente metricas
con `NaN`, metricas incompletas, payload incompleto y conteos incorrectos.
