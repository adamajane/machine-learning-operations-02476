"""API endpoints for triggering and monitoring training runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
import subprocess
import sys
from typing import Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
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


class TrainResponse(BaseModel):
    """Response returned when a training job is started."""

    job_id: str
    status_url: str


class TrainStatusResponse(BaseModel):
    """Response describing the current state of a training job."""

    job_id: str
    status: TrainStatus
    return_code: Optional[int]
    started_at: datetime


@dataclass
class TrainJob:
    """In-memory representation of a training job."""

    job_id: str
    process: subprocess.Popen[str]
    started_at: datetime


app = FastAPI(title="Rice CNN Classifier API")
_jobs: Dict[str, TrainJob] = {}


def _build_train_command(request: TrainRequest) -> list[str]:
    """Build the command used to launch a training job.

    Args:
        request: Training parameters from the client.

    Returns:
        Command list suitable for subprocess.Popen.
    """

    command = [
        sys.executable,
        "-m",
        "rice_cnn_classifier.train",
        "--epochs",
        str(request.epochs),
        "--batch-size",
        str(request.batch_size),
        "--learning-rate",
        str(request.learning_rate),
        "--data-path",
        request.data_path,
        "--model-dir",
        request.model_dir,
        "--wandb-project",
        request.wandb_project,
    ]

    if request.wandb_run_name:
        command.extend(["--wandb-run-name", request.wandb_run_name])

    if request.disable_wandb:
        command.append("--disable-wandb")

    return command


def _get_job_status(job: TrainJob) -> TrainStatusResponse:
    """Return the current status of a job.

    Args:
        job: The job to inspect.

    Returns:
        Status response describing the job.
    """

    return_code = job.process.poll()
    if return_code is None:
        status = TrainStatus.running
    elif return_code == 0:
        status = TrainStatus.succeeded
    else:
        status = TrainStatus.failed

    return TrainStatusResponse(
        job_id=job.job_id,
        status=status,
        return_code=return_code,
        started_at=job.started_at,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def start_training(request: TrainRequest) -> TrainResponse:
    """Start a new training job.

    Args:
        request: Training configuration supplied by the client.

    Returns:
        Metadata about the newly created training job.
    """

    command = _build_train_command(request)
    process = subprocess.Popen(command)
    job_id = str(uuid4())
    job = TrainJob(job_id=job_id, process=process, started_at=datetime.now(tz=UTC))
    _jobs[job_id] = job
    return TrainResponse(job_id=job_id, status_url=f"/train/{job_id}")


@app.get("/train/{job_id}", response_model=TrainStatusResponse)
def get_training_status(job_id: str) -> TrainStatusResponse:
    """Get the status of a training job.

    Args:
        job_id: ID of the job to inspect.

    Returns:
        Status information for the requested job.

    Raises:
        HTTPException: If the job does not exist.
    """

    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return _get_job_status(job)
