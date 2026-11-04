from datetime import datetime, timezone
from fastapi import APIRouter
from core.models import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check & provider status",
)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="AvatarSync Video & Audio Studio",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
