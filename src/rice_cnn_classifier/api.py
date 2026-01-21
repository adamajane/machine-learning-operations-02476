"""API endpoints for triggering training runs."""

from __future__ import annotations

from datetime import UTC, datetime
import os
import re
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest
from pydantic import BaseModel
import requests
import yaml


class TrainRequest(BaseModel):
    """Training request to launch a Vertex AI job."""

    display_name: Optional[str] = None
    project_id: Optional[str] = None
    region: Optional[str] = None


class TrainResponse(BaseModel):
    """Response returned when a training job is started."""

    job_name: str


app = FastAPI(title="Rice CNN Classifier API")


def _resolve_setting(value: Optional[str], env_name: str) -> str:
    """Resolve a setting from request or environment.

    Args:
        value: Value provided by the client.
        env_name: Environment variable name to check.

    Returns:
        Resolved value.

    Raises:
        HTTPException: If the setting is missing.
    """

    resolved = value or os.getenv(env_name)
    if not resolved:
        raise HTTPException(status_code=400, detail=f"Missing setting '{env_name}'.")
    return resolved


def _replace_env_vars(raw_text: str) -> str:
    """Replace ${VAR} placeholders with environment values.

    Args:
        raw_text: Raw config text.

    Returns:
        Config text with placeholders substituted.
    """

    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def repl(match: re.Match[str]) -> str:
        return os.getenv(match.group(1), "")

    return pattern.sub(repl, raw_text)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load the Vertex AI config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config data.
    """

    with open(config_path, "r", encoding="utf-8") as handle:
        raw_text = handle.read()
    rendered = _replace_env_vars(raw_text)
    return yaml.safe_load(rendered)


def _build_custom_job(config: Dict[str, Any], display_name: str) -> Dict[str, Any]:
    """Build a Vertex AI custom job payload.

    Args:
        config: Parsed config data.
        display_name: Job display name.

    Returns:
        Custom job payload.
    """

    if "jobSpec" in config:
        job_spec = config["jobSpec"]
    else:
        job_spec = config
    return {"displayName": display_name, "jobSpec": job_spec}


def _get_access_token() -> str:
    """Get a Google Cloud access token for the service account.

    Returns:
        Access token string.
    """

    credentials, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(GoogleAuthRequest())
    return credentials.token


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def start_training(request: TrainRequest) -> TrainResponse:
    """Start a new training job on Vertex AI using config_gpu.yaml.

    Args:
        request: Training configuration supplied by the client.

    Returns:
        Metadata about the newly created training job.
    """

    project_id = _resolve_setting(request.project_id, "VERTEX_PROJECT_ID")
    region = _resolve_setting(request.region, "VERTEX_REGION")
    config_path = os.getenv("VERTEX_JOB_CONFIG_PATH", "config_gpu.yaml")
    config = _load_config(config_path)
    display_name = request.display_name or f"rice-train-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
    payload = _build_custom_job(config, display_name)

    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/customJobs"
    headers = {"Authorization": f"Bearer {_get_access_token()}", "Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    job_name = response.json().get("name")
    if not job_name:
        raise HTTPException(status_code=500, detail="Vertex AI did not return a job name.")
    return TrainResponse(job_name=job_name)
