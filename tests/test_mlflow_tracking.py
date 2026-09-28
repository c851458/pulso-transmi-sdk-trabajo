from pathlib import Path

from src import mlflow_tracking


def test_dagshub_uri_is_derived_when_no_mlflow_uri(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("DAGSHUB_REPO_OWNER", "@example-owner")
    monkeypatch.setenv("DAGSHUB_REPO_NAME", "example-repo")

    assert mlflow_tracking._tracking_uri() == (
        "https://dagshub.com/example-owner/example-repo.mlflow"
    )


def test_dagshub_configuration_is_read_from_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DAGSHUB_REPO_OWNER", raising=False)
    monkeypatch.delenv("DAGSHUB_REPO_NAME", raising=False)
    (tmp_path / ".env").write_text(
        "DAGSHUB_REPO_OWNER=dotenv-owner\nDAGSHUB_REPO_NAME=dotenv-repo\n"
    )

    assert mlflow_tracking._tracking_uri() == (
        "https://dagshub.com/dotenv-owner/dotenv-repo.mlflow"
    )


def test_explicit_mlflow_uri_has_priority(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("DAGSHUB_REPO_OWNER", "example-owner")
    monkeypatch.setenv("DAGSHUB_REPO_NAME", "example-repo")

    assert mlflow_tracking._tracking_uri() == "https://mlflow.example.test"


def test_load_metadata_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert mlflow_tracking.load_metadata(tmp_path) == {}


def test_local_fallback_uses_sqlite_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("MLFLOW_TRACKING_URI", "DAGSHUB_REPO_OWNER", "DAGSHUB_REPO_NAME"):
        monkeypatch.delenv(name, raising=False)

    assert mlflow_tracking._tracking_uri() == f"sqlite:///{tmp_path / 'mlflow.db'}"
