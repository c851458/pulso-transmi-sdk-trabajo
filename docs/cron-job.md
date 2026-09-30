# Disparo de workflows con cron-job.org

GitHub no garantiza la frecuencia de `schedule` (en la práctica corre con horas
de retraso), así que los workflows se disparan desde
[cron-job.org](https://cron-job.org) mediante la API de `workflow_dispatch`.

| Job en cron-job.org | Workflow | Frecuencia |
| --- | --- | --- |
| `proyecto visu2` (existente) | `pipeline.yml` | Cada 10 min (:00, :10, …) |
| `pulso model-drift` (opcional) | `model_drift.yml` | Cada hora en el minuto 5 |

`pipeline.yml` no tiene `schedule` propio para no duplicar ejecuciones. Si se
crea el job de drift, hay que quitar también el `schedule` de `model_drift.yml`.

## 1. Token de GitHub

GitHub → Settings → Developer settings → Personal access tokens →
Fine-grained tokens → Generate new token:

- Repository access: *Only select repositories* →
  `c851458/pulso-transmi-sdk-trabajo`.
- Repository permissions: **Actions: Read and write**.

El token solo se guarda en cron-job.org; nunca en el repositorio.

## 2. Crear el job

| Campo | Valor |
| --- | --- |
| URL | `https://api.github.com/repos/c851458/pulso-transmi-sdk-trabajo/actions/workflows/<workflow>.yml/dispatches` |
| Method | `POST` |
| Headers | `Accept: application/vnd.github+json` · `Authorization: Bearer <token>` · `X-GitHub-Api-Version: 2022-11-28` · `Content-Type: application/json` |
| Body | `{"ref":"main"}` |
| Notifications | Avisar cuando la ejecución falle |

Horario: los workflows comparten el grupo de concurrencia
`pulso-transmi-supabase-writer`, que guarda **una sola** ejecución en espera;
si llega otra, GitHub cancela la que esperaba. El pipeline tarda 1-2 min y el
drift unos 4 min, por eso el drift se programa en el minuto 5.

## 3. Probar

*Test run* en cron-job.org:

| Respuesta | Significado |
| --- | --- |
| `204` | GitHub aceptó el disparo |
| `401` | Token inválido o vencido |
| `403` | Falta el permiso Actions: Read and write |
| `404` | URL, repositorio o workflow incorrectos, o el token no tiene acceso |
| `422` | Body inválido o rama inexistente |

## 4. Ver qué se ejecuta y qué no

1. **cron-job.org → History**: cada `204` es un disparo enviado. Un hueco
   significa que cron-job.org no disparó.
2. **GitHub → Actions** (filtrar por workflow):
   - verde: se ejecutó y publicó;
   - rojo con `PIPELINE WAITING_FOR_OPEN_CYCLE` (código 2): se ejecutó, pero la
     API no tenía ciclo abierto; es esperado;
   - rojo con otro error: fallo real;
   - gris *Cancelled*: reemplazado en la cola de concurrencia;
   - sin ejecución: el disparo no llegó (ver History).
3. **Log del pipeline**: `Recent accuracy …%` y luego `Skipping retraining: …`
   (publica con el modelo activo) o `Retraining: …` (reentrena). En el drift,
   se ejecuta el paso *Retrain and publish after drift* solo si decidió
   reentrenar.
4. **Terminal**:

   ```bash
   gh run list --workflow pipeline.yml --limit 20
   gh run list --workflow model_drift.yml --limit 20
   ```

5. **Dashboard**: la franja de reentrenamientos y el historial de drift
   muestran lo que llegó a Supabase.
