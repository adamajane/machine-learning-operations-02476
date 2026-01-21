"""API endpoints for triggering and monitoring training runs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from google.cloud import aiplatform_v1
from pydantic import BaseModel, Field


class TrainStatus(str, Enum):
    """Status values for training jobs."""

    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class TrainRequest(BaseModel):
    """Training parameters supplied by the API client."""

    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    data_path: str = Field(default="data/processed")
    model_dir: str = Field(default="models")
    wandb_project: str = Field(default="rice_cnn_classifier")
    wandb_run_name: Optional[str] = Field(default=None)
    disable_wandb: bool = Field(default=False)
    project_id: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)
    image_uri: Optional[str] = Field(default=None)
    machine_type: str = Field(default="n1-standard-8")
    accelerator_type: Optional[str] = Field(default=None)
    accelerator_count: int = Field(default=0, ge=0)


class TrainResponse(BaseModel):
    """Response returned when a training job is started."""

    job_id: str
    job_name: str
    status_url: str


class TrainStatusResponse(BaseModel):
    """Response describing the current state of a training job."""

    job_id: str
    job_name: str
    status: TrainStatus
    state: str
    created_at: Optional[datetime]


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


def _build_train_args(request: TrainRequest) -> list[str]:
    """Build the training arguments passed to the container.

    Args:
        request: Training configuration supplied by the client.

    Returns:
        List of CLI arguments for the training container.
    """

    args = [
        f"--epochs={request.epochs}",
        f"--batch-size={request.batch_size}",
        f"--learning-rate={request.learning_rate}",
        f"--data-path={request.data_path}",
        f"--model-dir={request.model_dir}",
        f"--wandb-project={request.wandb_project}",
    ]

    if request.wandb_run_name:
        args.append(f"--wandb-run-name={request.wandb_run_name}")

    if request.disable_wandb:
        args.append("--disable-wandb")

    return args


def _map_job_state(state: aiplatform_v1.JobState) -> TrainStatus:
    """Map a Vertex AI job state to the API status.

    Args:
        state: Vertex AI job state value.

    Returns:
        Simplified training status.
    """

    if state in {
        aiplatform_v1.JobState.JOB_STATE_QUEUED,
        aiplatform_v1.JobState.JOB_STATE_PENDING,
    }:
        return TrainStatus.queued
    if state == aiplatform_v1.JobState.JOB_STATE_RUNNING:
        return TrainStatus.running
    if state == aiplatform_v1.JobState.JOB_STATE_SUCCEEDED:
        return TrainStatus.succeeded
    return TrainStatus.failed


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def start_training(request: TrainRequest) -> TrainResponse:
    """Start a new training job on Vertex AI.

    Args:
        request: Training configuration supplied by the client.

    Returns:
        Metadata about the newly created training job.
    """

    project_id = _resolve_setting(request.project_id, "VERTEX_PROJECT_ID")
    region = _resolve_setting(request.region, "VERTEX_REGION")
    image_uri = _resolve_setting(request.image_uri, "TRAIN_IMAGE_URI")
    parent = f"projects/{project_id}/locations/{region}"
    job_client = aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )

    env_vars = [{"name": "PYTHONUNBUFFERED", "value": "1"}]
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key and not request.disable_wandb:
        env_vars.append({"name": "WANDB_API_KEY", "value": wandb_api_key})

    machine_spec = {"machine_type": request.machine_type}
    if request.accelerator_type and request.accelerator_count > 0:
        machine_spec["accelerator_type"] = request.accelerator_type
        machine_spec["accelerator_count"] = request.accelerator_count

    training_job = {
        "display_name": f"rice-train-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}",
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": machine_spec,
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": image_uri,
                        "args": _build_train_args(request),
                        "env": env_vars,
                    },
                }
            ]
        },
    }

    response = job_client.create_custom_job(parent=parent, custom_job=training_job)
    job_id = response.name.split("/")[-1]
    return TrainResponse(job_id=job_id, job_name=response.name, status_url=f"/train/{job_id}")


@app.get("/train/{job_id}", response_model=TrainStatusResponse)
def get_training_status(
    job_id: str, project_id: Optional[str] = None, region: Optional[str] = None
) -> TrainStatusResponse:
    """Get the status of a training job.

    Args:
        job_id: Vertex AI custom job resource name or job ID.
        project_id: GCP project ID override.
        region: Vertex AI region override.

    Returns:
        Status information for the requested job.
    """

    if "/" in job_id:
        parts = job_id.split("/")
        if len(parts) < 6:
            raise HTTPException(status_code=400, detail="Job name must be a full Vertex AI resource name.")
        resolved_region = parts[3]
        resolved_project = parts[1]
        resolved_name = job_id
        resolved_job_id = parts[-1]
    else:
        resolved_project = _resolve_setting(project_id, "VERTEX_PROJECT_ID")
        resolved_region = _resolve_setting(region, "VERTEX_REGION")
        resolved_name = f"projects/{resolved_project}/locations/{resolved_region}/customJobs/{job_id}"
        resolved_job_id = job_id

    job_client = aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": f"{resolved_region}-aiplatform.googleapis.com"}
    )
    job = job_client.get_custom_job(name=resolved_name)
    created_at = job.create_time.ToDatetime(tzinfo=UTC) if job.create_time else None
    return TrainStatusResponse(
        job_id=resolved_job_id,
        job_name=job.name,
        status=_map_job_state(job.state),
        state=job.state.name,
        created_at=created_at,
    )
