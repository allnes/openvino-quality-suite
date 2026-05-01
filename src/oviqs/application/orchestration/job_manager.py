from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["queued", "running", "pass", "warning", "fail", "unknown"]


@dataclass
class EvaluationJob:
    id: str
    kind: str
    status: JobStatus = "queued"
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, EvaluationJob] = {}

    def create(self, job: EvaluationJob) -> EvaluationJob:
        self._jobs[job.id] = job
        return job

    def update_status(self, job_id: str, status: JobStatus) -> EvaluationJob:
        job = self._jobs[job_id]
        job.status = status
        return job

    def get(self, job_id: str) -> EvaluationJob:
        return self._jobs[job_id]
