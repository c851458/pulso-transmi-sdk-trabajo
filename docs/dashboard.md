# Dashboard MLOps

El dashboard está implementado en `app/` como una aplicación Next.js de solo
lectura. No modifica el modelo, no ejecuta drift y no crea tablas.

## Datos usados

El endpoint `GET /api/dashboard` consulta las tablas existentes:

- `model` y `training_run`: modelo activo, versiones, entrenamiento y métricas JSONB.
- `model_drift_check`: serie de evaluaciones y rendimiento.
- `model_drift_execution`: auditoría de ejecuciones, PSI, ventanas, resultado y errores.
- `pipeline_execution` y `pipeline_sync_status`: estado complementario del pipeline.

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
