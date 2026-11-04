import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import UploadFile

from config import settings
from core.extractor import MediaExtractor
from core.logger import setup_logger
from core.manager import TTSManager, VideoManager
from core.models import APIResponse, JobData

logger = setup_logger("service.avatar")

ALLOWED_EXTENSIONS = {
    "wav", "mp3", "m4a", "aac", "flac", "ogg",
    "mp4", "mov", "avi", "webm", "mkv",
    "jpg", "jpeg", "png", "webp", "bmp",
}


class AvatarService:
    """Orchestrates end-to-end media extraction, voice cloning, and video lip-syncing."""

    def __init__(
        self,
        tts_manager: Optional[TTSManager] = None,
        video_manager: Optional[VideoManager] = None,
        extractor: Optional[MediaExtractor] = None,
    ):
        self.tts_manager = tts_manager or TTSManager()
        self.video_manager = video_manager or VideoManager()
        self.extractor = extractor or MediaExtractor()

        base_dir = Path(__file__).resolve().parent.parent
        self.upload_dir = base_dir / settings.upload_dir
        self.output_dir = base_dir / settings.output_dir
        self.temp_dir = base_dir / settings.temp_dir

        for d in [self.upload_dir, self.output_dir, self.temp_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, dict] = {}

    @staticmethod
    def is_allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_job(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    async def save_uploaded_media(self, video: UploadFile, job_id: str) -> str:
        if not video.filename or not self.is_allowed_file(video.filename):
            raise ValueError(f"Unsupported file format '{video.filename}'.")

        destination = self.upload_dir / f"{job_id}_{video.filename}"
        content = await video.read()
        with open(destination, "wb") as f:
            f.write(content)
        return str(destination)

    async def generate_avatar(
        self,
        script: str,
        video: Optional[UploadFile] = None,
        voice: Optional[str] = None,
        isolate_vocals: bool = False,
        enhance: bool = True,
        generate_video: bool = True,
    ) -> APIResponse:
        job_id = uuid.uuid4().hex[:8]
        start_time = time.time()
        logger.info("Received generation request. Assigned job_id: %s", job_id)

        script_clean = script.strip()
        if not script_clean:
            raise ValueError("Script text cannot be empty.")

        raw_video_path: Optional[str] = None
        if video and video.filename:
            raw_video_path = await self.save_uploaded_media(video, job_id)

        video_source: Optional[str] = raw_video_path if (raw_video_path and Path(raw_video_path).exists()) else None
        voice_clean = voice.strip() if voice and voice.strip() else None
        ref_audio_source: Optional[str] = raw_video_path if raw_video_path else None

        # Auto-select language-specific neural voice when speaker reference is absent
        if not voice_clean and not ref_audio_source:
            has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in script_clean)
            voice_clean = "hi-IN-MadhurNeural" if has_devanagari else "en-IN-PrabhatNeural"
            logger.info("No reference audio provided. Using natural preset: %s", voice_clean)

        clean_ref_audio = str(self.temp_dir / f"{job_id}_ref.wav")
        generated_speech = str(self.temp_dir / f"{job_id}_speech.mp3")
        extracted_audio: Optional[str] = None

        try:
            # Extract and filter reference audio if input media provided
            if ref_audio_source:
                try:
                    extracted_audio = self.extractor.extract_audio(
                        ref_audio_source,
                        clean_ref_audio,
                        max_duration_sec=15,
                        isolate_vocals=isolate_vocals,
                    )
                except Exception as audio_err:
                    logger.warning("Job %s: Audio extraction notice: %s", job_id, audio_err)
                    extracted_audio = ref_audio_source

            # Speech synthesis via fallback manager
            logger.info("Job %s: Synthesizing speech...", job_id)
            await self.tts_manager.synthesize(
                text=script_clean,
                output_path=generated_speech,
                reference_audio_path=extracted_audio,
                voice=voice_clean,
                enhance=enhance,
            )

            audio_filename = f"{job_id}_audio.mp3"
            final_audio_path = str(self.output_dir / audio_filename)
            Path(generated_speech).rename(final_audio_path)
            audio_url = f"/api/v1/avatar/{audio_filename}"

            # Video generation (Wav2Lip lip-sync or FFmpeg fallback)
            video_url: Optional[str] = None
            if generate_video and video_source and Path(video_source).exists():
                video_filename = f"{job_id}_video.mp4"
                final_video_path = str(self.output_dir / video_filename)
                logger.info("Job %s: Generating lip-synced video from %s...", job_id, Path(video_source).name)
                await self.video_manager.generate(
                    video_path=video_source,
                    audio_path=final_audio_path,
                    output_path=final_video_path,
                )
                video_url = f"/api/v1/avatar/{video_filename}"
                logger.info("Job %s: Video generated at %s", job_id, video_url)

            processing_time = round(time.time() - start_time, 2)
            job_record = {
                "job_id": job_id,
                "status": "completed",
                "audio_url": audio_url,
                "video_url": video_url,
                "processing_time_sec": processing_time,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._jobs[job_id] = job_record

            # Cleanup temp scratch audio
            Path(clean_ref_audio).unlink(missing_ok=True)
            Path(generated_speech).unlink(missing_ok=True)

            message = (
                "Voice synthesized and avatar video generated."
                if video_url
                else "Voice synthesized successfully."
            )
            logger.info("Job %s finished in %ss: audio=%s video=%s", job_id, processing_time, audio_url, video_url)
            return APIResponse(status="success", data=JobData(**job_record), message=message)

        except Exception as err:
            logger.error("Job %s execution failed: %s", job_id, err, exc_info=True)
            raise
