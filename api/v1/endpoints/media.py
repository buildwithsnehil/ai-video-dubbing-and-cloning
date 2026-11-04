from pathlib import Path
import typing as t
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from config import settings
from core.logger import setup_logger
from core.models import APIResponse

router = APIRouter()
logger = setup_logger("api.media", log_file=settings.log_file)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = (BASE_DIR / settings.output_dir).resolve()
UPLOAD_DIR = (BASE_DIR / settings.upload_dir).resolve()

MediaType = t.Literal["uploaded", "generated"]

def _get_secure_media_path(media_type: MediaType, filename: str) -> Path:
    """Validate filename and resolved path stays within the intended directory."""
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename specified.",
        )

    base_dir = OUTPUT_DIR if media_type == "generated" else UPLOAD_DIR
    target_path = (base_dir / safe_name).resolve()

    if not target_path.is_relative_to(base_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid media file path.",
        )
    return target_path


@router.get(
    "/{type}/{filename}",
    tags=["Media Delivery"],
    summary="Stream or download generated or uploaded media",
)
async def stream_media(type: MediaType, filename: str):
    file_path = _get_secure_media_path(type, filename)
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media file '{filename}' not found in {type}.",
        )
    return FileResponse(file_path)


@router.delete(
    "/{type}/{filename}",
    response_model=APIResponse,
    tags=["Media Delivery"],
    summary="Remove uploaded or generated media",
)
async def remove_media(type: MediaType, filename: str):
    file_path = _get_secure_media_path(type, filename)
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media file '{filename}' not found in {type}.",
        )

    try:
        file_path.unlink()
        logger.info("Deleted %s media file: %s", type, filename)
    except OSError as err:
        logger.error("Failed to delete %s media file '%s': %s", type, filename, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete media file: {err}",
        )

    return APIResponse(
        status="success",
        message=f"Media file '{filename}' deleted successfully from {type}.",
    )