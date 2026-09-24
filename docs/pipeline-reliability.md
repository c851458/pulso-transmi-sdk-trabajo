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
- Reintentos limitados para timeout, red, 408, 429 y 5xx, con backoff
  `5s, 15s, 30s` y jitter.
- Sin reintentos para 400, 401, 403, 404 y otros errores permanentes.
- La respuesta debe contener `submission_id`, `id` u `operation_id` (o una
  confirmacion booleana), estado valido y, cuando la API lo entrega, exactamente
  el numero esperado de predicciones y el ciclo correcto.
- El pipeline solo escribe `SUCCESS` despues de confirmar API; de lo contrario
  escribe `FAILED` y devuelve codigo distinto de cero.
- `concurrency` evita ejecuciones simultaneas.
- Solo `model_drift.yml` tiene schedule de inferencia/publicacion. `pipeline.yml`
  conserva `workflow_dispatch` para pruebas manuales.

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
