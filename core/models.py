from typing import Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str


class JobData(BaseModel):
    job_id: str
    status: str
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    processing_time_sec: float
    created_at: str


class APIResponse(BaseModel):
    status: str
    data: Optional[JobData] = None
    message: Optional[str] = None
