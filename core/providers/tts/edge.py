import asyncio
from pathlib import Path
import edge_tts
from core.logger import setup_logger
from core.providers.tts.jugnu import VOCAL_MASTERING_FILTERS

logger = setup_logger("provider.tts.edge")


class EdgeTTSProvider:
    """High-quality zero-cost neural text-to-speech using Microsoft Edge TTS."""

    async def generate_speech(
        self,
        text: str,
        output_path: str,
        reference_audio_path: str | None = None,
        voice: str | None = None,
        **kwargs
    ) -> str:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not voice:
            # Check for Devanagari script characters
            has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in text)
            voice = "hi-IN-MadhurNeural" if has_devanagari else "en-US-ChristopherNeural"

        enhance = kwargs.get("enhance", True)
        logger.info("Synthesizing speech via Edge TTS (Voice: %s, enhance=%s)...", voice, enhance)

        if enhance:
            raw_temp = out_file.with_name(f"raw_edge_{out_file.name}")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(raw_temp))

            if out_file.suffix.lower() == ".mp3":
                cmd = [
                    "ffmpeg", "-y", "-i", str(raw_temp),
                    "-af", VOCAL_MASTERING_FILTERS,
                    "-acodec", "libmp3lame", "-q:a", "0",
                    str(out_file),
                ]
            else:
                cmd = [
                    "ffmpeg", "-y", "-i", str(raw_temp),
                    "-af", VOCAL_MASTERING_FILTERS,
                    "-acodec", "pcm_s16le", "-ar", "24000",
                    str(out_file),
                ]

            logger.info("Applying studio vocal mastering chain to Edge TTS audio...")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace")
                logger.warning("Vocal mastering on Edge TTS failed (%s); using raw audio.", err_msg)
                raw_temp.rename(out_file)
            else:
                raw_temp.unlink(missing_ok=True)
        else:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(out_file))

        logger.info("Speech audio written to: %s", output_path)
        return output_path
