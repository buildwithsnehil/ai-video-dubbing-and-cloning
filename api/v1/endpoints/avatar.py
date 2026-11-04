from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.deps import get_avatar_service
from core.models import APIResponse, JobData
from services.avatar_service import AvatarService

router = APIRouter()


@router.post(
    "/generate",
    response_model=APIResponse,
    tags=["Avatar Generation"],
    summary="Generate lip-synced avatar video from script text",
)
async def generate_speech(
    script: str = Form(..., description="Target speech script to synthesize"),
    video: Optional[UploadFile] = File(None, description="Optional speaker video (MP4, WEBM). If omitted, default preset voice is used."),
    voice: Optional[str] = Form(None, description="Optional preset voice ID (e.g., hi-IN-MadhurNeural, en-IN-PrabhatNeural)"),
    isolate_vocals: bool = Form(False, description="Enable AI vocal separation from noisy background audio"),
    enhance: bool = Form(True, description="Enable studio vocal presence EQ, compression, and loudness mastering"),
    generate_video: bool = Form(True, description="Generate lip-synced avatar video"),
    avatar_service: AvatarService = Depends(get_avatar_service),
):
    try:
        return await avatar_service.generate_avatar(
            script=script,
            video=video,
            voice=voice,
            isolate_vocals=isolate_vocals,
            enhance=enhance,
            generate_video=generate_video,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)
        )


@router.get(
    "/jobs/{job_id}",
    response_model=APIResponse,
    tags=["Avatar Generation"],
    summary="Get job status by ID",
)
async def get_job_status(
    job_id: str,
    avatar_service: AvatarService = Depends(get_avatar_service),
):
    job = avatar_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found."
        )
    return APIResponse(status="success", data=JobData(**job))
