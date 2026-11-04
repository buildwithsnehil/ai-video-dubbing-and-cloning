from fastapi import APIRouter

from api.v1.endpoints import avatar, health, media

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(avatar.router, prefix="/avatar")
api_router.include_router(media.router, prefix="/avatar")
