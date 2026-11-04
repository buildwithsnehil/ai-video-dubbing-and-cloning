from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.v1.router import api_router
from config import settings
from core.logger import setup_logger

logger = setup_logger("api.app", log_file=settings.log_file)

app = FastAPI(
    title="AvatarSync Video & Audio Studio",
    description="AI speech synthesis, voice cloning, and lip-synced avatar video generator.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount versioned API routes
app.include_router(api_router)


@app.get("/")
async def serve():
    return JSONResponse(content={"message": "Welcome to AvatarSync Studio"})