# Dashboard MLOps

El dashboard está implementado en `app/` como una aplicación Next.js de solo
lectura. No modifica el modelo, no ejecuta drift y no crea tablas.

## Datos usados

El endpoint `GET /api/dashboard` consulta las tablas existentes:

- `model` y `training_run`: modelo activo, versiones, entrenamiento y métricas JSONB.
- `model_drift_check`: serie de evaluaciones y rendimiento.
- `model_drift_execution`: auditoría de ejecuciones, PSI, ventanas, resultado y errores.
- `pipeline_execution` y `pipeline_sync_status`: estado complementario del pipeline.

Para las gráficas históricas se hacen dos consultas acotadas a los últimos 7
días: `training_run` solo con `metrics->test->{wape,mae,rmse,r2,accuracy}` (no el JSONB
completo) y `model_drift_check` con PSI, variables con drift, rendimiento y
decisión.

## Paneles

- **Métricas actuales**: WAPE, MAE, RMSE, R², accuracy y MAE de validación
  cruzada de la evaluación de prueba del modelo activo, con la variación
  frente al modelo anterior (▲ mejor / ▼ peor según la dirección de cada
  métrica), más el WAPE en producción y la **accuracy reciente** de la última
  evaluación de drift: la misma medida que decide el reentrenamiento, con el
  umbral (`RETRAIN_ACCURACY_THRESHOLD`, 79 %) y su origen (pronósticos frente
  a demanda observada, o evaluación en la ventana reciente).
- **Drift y reentrenamientos** (rango 24 h / 72 h / 7 días):
  - *WAPE: prueba al reentrenar vs. producción*: WAPE de prueba de cada modelo
    (escalonado, cambia en cada reentrenamiento) y WAPE reciente medido por el
    control de drift.
  - *Variables con drift*: número de variables con PSI sobre el umbral, con la
    línea de alerta (`drifted_features_required`) y un triángulo en cada alerta.
  - *Reentrenamientos*: una marca por modelo entrenado y, en rojo, las
    evaluaciones de drift que recomendaron reentrenar.
  - Tabla desplegable con los reentrenamientos del rango.
  - *Distribución de las métricas*: un histograma por métrica (WAPE, MAE,
    RMSE, R² y accuracy de prueba, una muestra por reentrenamiento; accuracy en
    producción, una muestra por evaluación de drift), con n, mediana, mínimo,
    máximo y una línea punteada en el valor del modelo activo. Si una métrica
    no varió en el rango se dibuja una sola barra con la leyenda
    "sin variación".

Las gráficas están en `app/charts.tsx` (SVG sin dependencias), con tooltip y
cursor al pasar el puntero.

Las métricas no presentes en esas tablas se muestran como `—`; el dashboard no
las estima. Los umbrales se leen del JSONB de la última evaluación y usan
`0.20`, `1.25` y `2` únicamente si el registro no los contiene.

## Vercel

1. Importa el repositorio en Vercel y conserva el framework detectado como Next.js.
2. Configura en Project Settings → Environment Variables:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY` (solo Server-side, no `NEXT_PUBLIC_`).
3. Ejecuta el despliegue con el build configurado en `vercel.json`.

La base de datos debe conservar RLS habilitado. El endpoint usa la service role
solo en el servidor para leer las tablas existentes; nunca se incluye esa clave
en el bundle del navegador. Si se prefiere usar la anon key, deben crearse
políticas de lectura explícitas en Supabase, fuera del alcance de este dashboard.

## Desarrollo local

```bash
npm install
npm run dev
```

El refresh automático ocurre cada 5 minutos y el botón `Actualizar` solo hace
una consulta GET. El endpoint consulta únicamente columnas necesarias y excluye
artefactos del modelo y respuestas JSON grandes de los listados. No existe
ninguna ruta POST ni acción que dispare el pipeline.

## Validación de disponibilidad

El build se valida con `npm run build`. La validación contra Supabase requiere
que el proyecto esté disponible y que sus variables de entorno sean válidas;
si Supabase no responde, la interfaz muestra el error sin inventar datos.
