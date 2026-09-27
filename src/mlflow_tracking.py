from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


METADATA_FILENAME = "mlflow_metadata.json"


def _config_value(name: str) -> str | None:
    """Read process environment first, then the local ignored .env file."""
    value = os.getenv(name)
    if value is not None:
        return value
    dotenv = Path(".env")
    if not dotenv.exists():
        return None
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name:
            return raw_value.strip().strip('"').strip("'")
    return None


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            values.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        values[prefix[:250]] = float(value)
    return values


def _safe_params(params: dict[str, Any]) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for key, value in params.items():
        text = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
        rendered[str(key)[:250]] = text[:500]
    return rendered


def _tracking_uri() -> str:
    uri = _config_value("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    owner = _config_value("DAGSHUB_REPO_OWNER")
    repository = _config_value("DAGSHUB_REPO_NAME")
    if owner and repository:
        return f"https://dagshub.com/{owner.lstrip('@')}/{repository}.mlflow"
    return Path("mlruns").resolve().as_uri()


def log_training_run(
    *,
    model: Any,
    metrics: dict[str, Any],
    artifact_dir: Path,
    params: dict[str, Any],
    tags: dict[str, str],
) -> dict[str, str]:
    """Log one completed training and register its model version.

    MLflow is imported lazily so the rest of the SDK remains importable in
    environments that do not execute training.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    tracking_uri = _tracking_uri()
    experiment_name = _config_value("MLFLOW_EXPERIMENT_NAME") or "pulso-transmi"
    registered_model_name = _config_value("MLFLOW_REGISTERED_MODEL_NAME") or "pulso-transmi-demand"
    username = (
        _config_value("MLFLOW_TRACKING_USERNAME")
        or _config_value("MLFLOW_USERNAME")
        or _config_value("DAGSHUB_REPO_OWNER")
    )
    password = (
        _config_value("MLFLOW_TRACKING_PASSWORD")
        or _config_value("MLFLOW_PASSWORD")
        or _config_value("DAGSHUB_TOKEN")
    )
    if username:
        username = username.lstrip("@")
    if username:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", username)
    if password:
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", password)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(_safe_params(params))
        mlflow.log_metrics(_flatten_numeric(metrics))
        for key, value in tags.items():
            mlflow.set_tag(key, str(value))
        for artifact in artifact_dir.iterdir():
            if artifact.is_file() and artifact.name != METADATA_FILENAME:
                mlflow.log_artifact(str(artifact), artifact_path="training_artifacts")
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            serialization_format="cloudpickle",
        )
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    matching = [version for version in versions if version.run_id == run_id]
    if not matching:
        raise RuntimeError(
            f"MLflow registered no model version for run {run_id}"
        )
    model_version = str(max(matching, key=lambda item: int(item.version)).version)
    client.set_tag(run_id, "mlflow_model_name", registered_model_name)
    client.set_tag(run_id, "mlflow_model_version", model_version)
    metadata = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
        "run_id": run_id,
        "model_version": model_version,
        "model_uri": model_info.model_uri,
    }
    (artifact_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_metadata(artifact_dir: Path) -> dict[str, str]:
    path = artifact_dir / METADATA_FILENAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_registered_model(model_name: str, model_version: str) -> Any:
    import mlflow.sklearn

    mlflow.set_tracking_uri(_tracking_uri())
    return mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
