"""API endpoints for triggering training runs."""

from __future__ import annotations

from datetime import UTC, datetime
import base64
import os
import re
from typing import Any, Dict, Optional, Mapping

from fastapi import Body, FastAPI, HTTPException
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


def _replace_env_vars(raw_text: str, overrides: Mapping[str, str]) -> str:
    """Replace ${VAR} placeholders with environment values.

    Args:
        raw_text: Raw config text.
        overrides: Explicit values to use before reading environment variables.

    Returns:
        Config text with placeholders substituted.
    """

    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in overrides:
            return overrides[key]
        return os.getenv(key, "")

    return pattern.sub(repl, raw_text)


def _load_config(config_path: str, overrides: Mapping[str, str]) -> Dict[str, Any]:
    """Load the Vertex AI config file.

    Args:
        config_path: Path to the YAML config file.
        overrides: Explicit values to use during template rendering.

    Returns:
        Parsed config data.
    """

    with open(config_path, "r", encoding="utf-8") as handle:
        raw_text = handle.read()
    rendered = _replace_env_vars(raw_text, overrides)
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


def _get_secret_value(project_id: str, secret_name: str) -> Optional[str]:
    """Fetch a secret value from Secret Manager.

    Args:
        project_id: GCP project id.
        secret_name: Secret name in Secret Manager.

    Returns:
        Secret value if available.
    """

    url = (
        "https://secretmanager.googleapis.com/v1/"
        f"projects/{project_id}/secrets/{secret_name}/versions/latest:access"
    )
    headers = {"Authorization": f"Bearer {_get_access_token()}"}
    response = requests.get(url, headers=headers, timeout=30)
    if not response.ok:
        return None
    payload = response.json()
    data = payload.get("payload", {}).get("data")
    if not data:
        return None
    if not isinstance(data, str):
        return None
    decoded = base64.b64decode(data)
    return decoded.decode("utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def start_training(request: TrainRequest | None = Body(default=None)) -> TrainResponse:
    """Start a new training job on Vertex AI using config_gpu.yaml.

    Args:
        request: Training configuration supplied by the client.

    Returns:
        Metadata about the newly created training job.
    """

    request = request or TrainRequest()
    project_id = _resolve_setting(request.project_id, "VERTEX_PROJECT_ID")
    region = _resolve_setting(request.region, "VERTEX_REGION")
    config_path = os.getenv("VERTEX_JOB_CONFIG_PATH", "config_gpu.yaml")
    wandb_secret_name = os.getenv("WANDB_SECRET_NAME", "WANDB_API_KEY")
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = handle.read()
    overrides: Dict[str, str] = {}
    if not os.getenv("WANDB_API_KEY"):
        secret_value = _get_secret_value(project_id, wandb_secret_name)
        if secret_value:
            overrides["WANDB_API_KEY"] = secret_value
    if "${WANDB_API_KEY}" in raw_config and "WANDB_API_KEY" not in overrides and not os.getenv("WANDB_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="Missing WANDB_API_KEY and Secret Manager lookup failed.",
        )
    config = _load_config(config_path, overrides)
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
