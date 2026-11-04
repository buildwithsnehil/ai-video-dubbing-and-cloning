import asyncio
from pathlib import Path
from core.logger import setup_logger

logger = setup_logger("provider.video.ffmpeg")


class FFmpegVideoProvider:
    """Local fallback video generator combining reference frame/video and speech audio using FFmpeg."""

    async def generate_video(
        self, video_path: str, audio_path: str, output_path: str, **kwargs
    ) -> str:
        input_video = video_path or kwargs.get("image_path")
        out_p = Path(output_path).resolve()
        vid_p = Path(input_video).resolve()
        aud_p = Path(audio_path).resolve()

        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not vid_p.exists():
            raise FileNotFoundError(f"Source video not found: {vid_p}")
        if not aud_p.exists():
            raise FileNotFoundError(f"Synthesized audio source not found: {aud_p}")

        logger.info("Assembling fallback video using local FFmpeg...")

        cmd = [
            "ffmpeg",
            "-y",
            "-stream_loop", "-1",
            "-i", str(vid_p),
            "-i", str(aud_p),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(out_p),
        ]

        logger.debug("FFmpeg assembly command: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        logger.debug("FFmpeg assembly returncode=%s", proc.returncode)
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")
            logger.error("FFmpeg video assembly failed: %s", err_msg)
            raise RuntimeError(f"FFmpeg assembly failed: {err_msg}")

        logger.info("Fallback video written to: %s", out_p)
        return str(out_p)
