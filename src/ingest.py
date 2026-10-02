from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from pulso_transmi import PulsoTransmiClient
from src.supabase_resilience import SupabaseRestClient


LOG = logging.getLogger("pulso.ingest")
BATCH_SIZE = 500
OVERLAP = timedelta(minutes=30)
_SYNC_STATUS_AVAILABLE = True


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        if key.startswith(("SUPABASE_", "PULSO_")):
            values[key] = value
    return values


def chunks(rows: list[dict[str, Any]], size: int = BATCH_SIZE):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def update_sync_status(
    db: SupabaseRestClient,
    *,
    status: str,
    started_at: str | None = None,
    records_received: int = 0,
    records_inserted: int = 0,
    records_updated: int = 0,
    records_unchanged: int = 0,
    error: str | None = None,
) -> None:
    """Publish a safe, frontend-readable heartbeat for the latest ingestion."""
    global _SYNC_STATUS_AVAILABLE
    if not _SYNC_STATUS_AVAILABLE:
        return
    now = pd.Timestamp.utcnow().isoformat()
    try:
        previous = db.rows("pipeline_sync_status", select="last_updated_at", id="eq.singleton", limit="1")
        last_updated_at = now if status == "SUCCESS" else (previous[0].get("last_updated_at") if previous else None)
        db.upsert_many("pipeline_sync_status", [{
            "id": "singleton",
            "status": status,
            "started_at": started_at,
            "last_updated_at": last_updated_at,
            "records_received": records_received,
            "records_inserted": records_inserted,
            "records_updated": records_updated,
            "records_unchanged": records_unchanged,
            "error": error,
            "updated_at": now,
        }], "id")
    except RuntimeError as exc:
        if "404" in str(exc):
            _SYNC_STATUS_AVAILABLE = False
            LOG.warning("pipeline_sync_status no existe aún; se continúa sin heartbeat hasta aplicar la migración")
            return
        raise


def latest_timestamp(db: SupabaseRestClient, table: str) -> pd.Timestamp | None:
    rows = db.rows(table, select="observed_at", order="observed_at.desc", limit="1")
    if not rows:
        return None
    return pd.Timestamp(rows[0]["observed_at"])


def ensure_data_cut(db: SupabaseRestClient, api: PulsoTransmiClient, env: dict[str, str]) -> int:
    existing = db.rows("data_cut", select="id", order="queried_at.desc", limit="1")
    if existing:
        return int(existing[0]["id"])
    meta = api.meta()
    dataset = meta["dataset"]
    source = db.rows("data_source", select="id", name="eq.Pulso TransMi API", limit="1")
    if source:
        source_id = int(source[0]["id"])
    else:
        source_id = int(db.insert_many("data_source", [{
            "name": "Pulso TransMi API",
            "source_type": "http_api",
            "base_url": env["PULSO_API_URL"],
        }])[0]["id"])
    cut = db.insert_many("data_cut", [{
        "source_id": source_id,
        "dataset_version": dataset["dataset"],
        "period_start": dataset["history_start"],
        "period_end": dataset["history_end"],
        "metadata_sha256": dataset["files"]["observations.csv"]["sha256"],
        "code_commit": os.getenv("GITHUB_SHA"),
    }])
    return int(cut[0]["id"])


def sync_stations(db: SupabaseRestClient, stations: pd.DataFrame) -> int:
    rows = stations.fillna("").to_dict("records")
    payload = [{
        "station_id": str(row["station_id"]).zfill(5),
        "station_name": row.get("station_name", row.get("name", "")),
        "corridor": row["corridor"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
    } for row in rows]
    count = 0
    for batch in chunks(payload):
        count += len(db.upsert_many("station", batch, "station_id"))
    return count


def combine_observations(starter: pd.DataFrame, stream: pd.DataFrame, start: pd.Timestamp | None) -> pd.DataFrame:
    """Starter and competition-stream observations from ``start`` on; the stream wins on repeated keys.

    Older rows are already stored, and keeping them would make every run re-read the whole stream.
    """
    columns = ["station_id", "observed_at", "demand"]
    frames = [frame[columns] for frame in (starter, stream) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    if start is not None:
        combined = combined[combined["observed_at"] >= start]
    combined["station_id"] = combined["station_id"].astype(str).str.zfill(5)
    return combined.drop_duplicates(["station_id", "observed_at"], keep="last").reset_index(drop=True)


def sync_observations(db: SupabaseRestClient, table: str, data_cut_id: int, frame: pd.DataFrame, columns: list[str]) -> tuple[int, int, int]:
    if frame.empty:
        return 0, 0, 0
    start = frame["observed_at"].min().isoformat()
    existing = db.rows(
        table,
        select=",".join(columns),
        data_cut_id=f"eq.{data_cut_id}",
        observed_at=f"gte.{start}",
        limit="10000",
    )
    existing_by_key = {(str(row["station_id"]), str(row["observed_at"])): row for row in existing if "station_id" in row}
    inserts: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    ignored = 0
    for row in frame.to_dict("records"):
        station_id = str(row["station_id"]).zfill(5)
        observed_at = pd.Timestamp(row["observed_at"]).isoformat()
        values = {column: row[column] for column in columns if column not in ("station_id", "observed_at")}
        key = (station_id, observed_at)
        if key not in existing_by_key:
            inserts.append({"data_cut_id": data_cut_id, "station_id": station_id, "observed_at": observed_at, **values})
        elif any(str(existing_by_key[key].get(column)) != str(value) for column, value in values.items()):
            updates.append({"data_cut_id": data_cut_id, "station_id": station_id, "observed_at": observed_at, **values})
        else:
            ignored += 1
    for batch in chunks(inserts):
        db.insert_many(table, batch)
    for batch in chunks(updates):
        db.upsert_many(table, batch, "data_cut_id,station_id,observed_at")
    LOG.info("%s: %d nuevos, %d actualizados, %d duplicados ignorados", table, len(inserts), len(updates), ignored)
    return len(inserts), len(updates), ignored


def sync_context(db: SupabaseRestClient, data_cut_id: int, frame: pd.DataFrame) -> tuple[int, int, int]:
    if frame.empty:
        return 0, 0, 0
    start = frame["observed_at"].min().isoformat()
    existing = db.rows(
        "context_observation",
        select="observed_at,event_intensity,rain_forecast,rain_mm,temperature_c,temperature_forecast",
        data_cut_id=f"eq.{data_cut_id}",
        observed_at=f"gte.{start}",
        limit="10000",
    )
    existing_by_key = {str(row["observed_at"]): row for row in existing}
    value_columns = ["event_intensity", "rain_forecast", "rain_mm", "temperature_c", "temperature_forecast"]
    inserts: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    ignored = 0
    for row in frame.to_dict("records"):
        observed_at = pd.Timestamp(row["observed_at"]).isoformat()
        values = {column: row[column] for column in value_columns}
        if observed_at not in existing_by_key:
            inserts.append({"data_cut_id": data_cut_id, "observed_at": observed_at, **values})
        elif any(str(existing_by_key[observed_at].get(column)) != str(value) for column, value in values.items()):
            updates.append({"data_cut_id": data_cut_id, "observed_at": observed_at, **values})
        else:
            ignored += 1
    for batch in chunks(inserts):
        db.insert_many("context_observation", batch)
    for batch in chunks(updates):
        db.upsert_many("context_observation", batch, "data_cut_id,observed_at")
    LOG.info("context_observation: %d nuevos, %d actualizados, %d duplicados ignorados", len(inserts), len(updates), ignored)
    return len(inserts), len(updates), ignored


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("Inicio de extracción incremental")
    env = load_env()
    required = ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "PULSO_API_URL", "PULSO_API_KEY")
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise RuntimeError(f"Faltan variables requeridas: {', '.join(missing)}")
    with SupabaseRestClient(env["SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"]) as db:
        started_at = pd.Timestamp.utcnow().isoformat()
        try:
            update_sync_status(db, status="SYNCING", started_at=started_at)
            with PulsoTransmiClient(base_url=env["PULSO_API_URL"], api_key=env["PULSO_API_KEY"]) as api:
                LOG.info("Conexión configurada con la API de Pulso TransMi")
                last_demand = latest_timestamp(db, "demand_observation")
                last_context = latest_timestamp(db, "context_observation")
                start_values = [value for value in (last_demand, last_context) if value is not None]
                start = min(start_values) - OVERLAP if start_values else None
                start_text = start.isoformat() if start is not None else None
                LOG.info("Extrayendo API desde %s", start_text or "el inicio disponible")
                stations = api.stations()
                observations = api.observations_dataframe(start=start_text)
                # /v1/observations ends with the starter dataset; newer demand only arrives through the
                # stream. Context stopped there too, so the stream starts from the latest demand alone.
                stream_start = last_demand - OVERLAP if last_demand is not None else None
                stream = api.stream_observations_dataframe()
                LOG.info("Stream de competencia: %d registros; se sincronizan desde %s", len(stream), stream_start)
                observations = combine_observations(observations, stream, stream_start)
                context = api.context_dataframe(start=start_text)
                received = len(stations) + len(observations) + len(context)
                LOG.info("Registros obtenidos: estaciones=%d demanda=%d contexto=%d", len(stations), len(observations), len(context))
                cut_id = ensure_data_cut(db, api, env)
                station_count = sync_stations(db, stations)
                demand_counts = sync_observations(db, "demand_observation", cut_id, observations, ["station_id", "observed_at", "demand"])
                context_counts = sync_context(db, cut_id, context)
                inserted = station_count + demand_counts[0] + context_counts[0]
                updated = demand_counts[1] + context_counts[1]
                unchanged = demand_counts[2] + context_counts[2]
                LOG.info("Estaciones procesadas: %d", station_count)
                LOG.info(
                    "Resultado Supabase: demanda nuevos=%d actualizados=%d ignorados=%d; contexto nuevos=%d actualizados=%d ignorados=%d",
                    *demand_counts,
                    *context_counts,
                )
                update_sync_status(
                    db,
                    status="SUCCESS",
                    started_at=started_at,
                    records_received=received,
                    records_inserted=inserted,
                    records_updated=updated,
                    records_unchanged=unchanged,
                )
            LOG.info("Sincronización completada; última actualización: %s", pd.Timestamp.utcnow().isoformat())
        except Exception as exc:
            try:
                update_sync_status(db, status="FAILED", started_at=started_at, error=f"{type(exc).__name__}: {exc}")
            except Exception:
                LOG.exception("No fue posible registrar el estado FAILED de la sincronización")
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Error de conexión o procesamiento; la siguiente ejecución podrá reintentarlo")
        LOG.error("Finalización con error")
        raise
