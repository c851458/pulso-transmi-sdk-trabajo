# Control de drift y reentrenamiento

La Action [model_drift.yml](../.github/workflows/model_drift.yml) ejecuta este
flujo cada diez minutos y también permite `workflow_dispatch`:

1. ingesta incremental de API a Supabase;
2. comparación de una ventana de referencia de 7 días contra los 7 días más
   recientes;
3. cálculo de PSI para demanda, eventos y variables meteorológicas;
4. comparación de WAPE reciente contra el WAPE del entrenamiento;
5. persistencia del diagnóstico en `public.model_drift_check`;
6. persistencia de la auditoría de cada ejecución en `public.model_drift_execution`;
7. reentrenamiento y publicación solo cuando la decisión sea `retrain=true`.

Si la publicación del modelo actual falla por una indisponibilidad transitoria
de la API o de Supabase, el job intenta publicar hasta tres veces, con esperas
de 60 y 120 segundos. Si todos fallan, la siguiente ejecución programada de
10 minutos vuelve a intentarlo. No existe un retry infinito.

## Umbrales por defecto

| Variable | Valor | Acción |
| --- | ---: | --- |
| `DRIFT_PSI_THRESHOLD` | `0.20` | Una variable se considera drifted desde este PSI |
| `DRIFTED_FEATURES_REQUIRED` | `2` | Variables drifted necesarias para activar alerta |
| `DRIFT_PERFORMANCE_RATIO` | `1.25` | WAPE reciente >= 1.25x el baseline |
| `DRIFT_MIN_ROWS` | `96` | Mínimo de filas en cada ventana |
| `RETRAIN_COOLDOWN_HOURS` | `24` | Evita reentrenamientos repetidos |

Los valores pueden configurarse como GitHub Actions Variables, no como Secrets,
porque no son credenciales. La Action usa los Secrets existentes para
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PULSO_API_URL` y `PULSO_API_KEY`.

En la ejecución validada del 2026-09-23 hubo 8.064 filas de referencia y 8.076
actuales. `event_intensity` tuvo PSI `3.8656`, pero la alerta global quedó en
`false` porque solo una variable superó el umbral y el cooldown estaba activo.

El diagnóstico se conserva con ventanas, PSI, WAPE, umbrales, modelo, decisión
y motivo en `model_drift_check`. La auditoría `model_drift_execution` conserva
además ejecuciones exitosas y fallidas, con su identificador, timestamps, estado
y mensaje de error. Un fallo marca el job como fallido y deja las siguientes
ejecuciones programadas disponibles para reintentar.
