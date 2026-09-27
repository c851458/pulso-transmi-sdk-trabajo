# Tracking y versionamiento con MLflow

El entrenamiento existente conserva su metodología y, al finalizar
correctamente, crea un Run de MLflow con el modelo seleccionado, sus métricas,
parámetros y artefactos. También registra una nueva versión en Model Registry.

## Configuración

Variables soportadas:

```text
MLFLOW_TRACKING_URI
MLFLOW_USERNAME
MLFLOW_PASSWORD
DAGSHUB_REPO_OWNER
DAGSHUB_REPO_NAME
DAGSHUB_TOKEN
MLFLOW_EXPERIMENT_NAME=pulso-transmi
MLFLOW_REGISTERED_MODEL_NAME=pulso-transmi-demand
MLFLOW_ENVIRONMENT=development
DATASET_VERSION=supabase-current
TRAINING_TYPE=baseline-or-drift-retrain
```

Si `MLFLOW_TRACKING_URI` no está definida y existen `DAGSHUB_REPO_OWNER` y
`DAGSHUB_REPO_NAME`, el código utiliza automáticamente:

```text
https://dagshub.com/<owner>/<repository>.mlflow
```

`DAGSHUB_TOKEN` se utiliza como contraseña de MLflow. En producción se deben
configurar `DAGSHUB_REPO_OWNER` y `DAGSHUB_REPO_NAME` como Variables de GitHub,
y `DAGSHUB_TOKEN` como Secret. `MLFLOW_TRACKING_URI` tiene prioridad si se
necesita usar otra instancia compatible.

Si no se define ninguna configuración remota, se utiliza tracking local en
`mlruns/`.

## Datos registrados

Cada Run registra:

- parámetros del modelo, predictores, objetivo, split cronológico, número de
  folds, semilla, tablas y ventana temporal;
- métricas CV, test, WAPE, accuracy, MAE, RMSE, R² y estabilidad;
- tags de proyecto, tipo de modelo, entorno, versión de datos, tipo de
  entrenamiento y commit;
- los artefactos actuales del baseline y el modelo sklearn serializado.

La relación con Supabase se guarda en `model.mlflow_run_id`,
`model.mlflow_model_name`, `model.mlflow_model_version` y
`training_run.mlflow_run_id`.

## Consultas y carga

Desde la interfaz de MLflow se puede seleccionar el experimento `pulso-transmi`
y consultar sus Runs. La versión registrada se puede cargar desde Python:

```python
from src.mlflow_tracking import load_registered_model

model = load_registered_model("pulso-transmi-demand", "1")
prediction = model.predict(features)
```

En Supabase:

```sql
select id, version, mlflow_run_id, mlflow_model_name,
       mlflow_model_version, status, trained_at
from public.model
order by trained_at desc;
```
