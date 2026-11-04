import asyncio
import time
from typing import Any
from config.config import settings
from core.logger import setup_logger
from core.providers import (
    EdgeTTSProvider,
    ElevenLabsProvider,
    JugnuTTSProvider,
    FFmpegVideoProvider,
    Wav2LipVideoProvider,
)

logger = setup_logger("core.manager")


class TTSManager:
    """Manages text-to-speech providers with automatic fallback chaining."""

    def __init__(self):
        self.providers: dict[str, Any] = {
            "jugnu": JugnuTTSProvider(),
            "edge_tts": EdgeTTSProvider(),
            "elevenlabs": ElevenLabsProvider(),
        }

    def get_provider_chain(self) -> list[tuple[str, Any]]:
        primary = settings.active_tts_provider.lower().strip()
        fallback_order = ["jugnu", "edge_tts", "elevenlabs"]
        chain_keys = [primary] + [k for k in fallback_order if k != primary]

        chain = []
        for key in chain_keys:
            if key in self.providers and (key, self.providers[key]) not in chain:
                chain.append((key, self.providers[key]))
        logger.debug("Resolved TTS provider chain: %s", [k for k, _ in chain])
        return chain

    def get_provider(self, name: str | None = None) -> Any:
        key = (name or settings.active_tts_provider).lower().strip()
        return self.providers.get(key, self.providers["edge_tts"])

    async def synthesize(
        self,
        text: str,
        output_path: str,
        reference_audio_path: str | None = None,
        voice: str | None = None,
        **kwargs
    ) -> str:
        errors: list[str] = []
        logger.debug(
            "Synthesize requested: text_len=%d, ref_audio=%s, voice=%s, output=%s",
            len(text),
            reference_audio_path,
            voice,
            output_path,
        )

        for name, provider in self.get_provider_chain():
            try:
                # Jugnu requires reference speaker audio for zero-shot cloning
                if name == "jugnu" and not reference_audio_path:
                    logger.debug("Skipping '%s': missing reference audio path.", name)
                    continue

                call_voice = voice
                if name == "edge_tts" and not call_voice:
                    has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in text)
                    call_voice = "hi-IN-MadhurNeural" if has_devanagari else "en-IN-PrabhatNeural"
                    logger.debug("Edge TTS auto-selected voice: %s", call_voice)

                logger.info("Attempting speech synthesis with provider: %s", name)
                t0 = time.time()
                result = await provider.generate_speech(
                    text=text,
                    output_path=output_path,
                    reference_audio_path=reference_audio_path,
                    voice=call_voice,
                    **kwargs
                )
                elapsed = round(time.time() - t0, 3)
                logger.debug("Provider '%s' completed synthesis in %ss -> %s", name, elapsed, result)
                return result
            except Exception as err:
                logger.warning("TTS provider '%s' failed: %s", name, err)
                logger.debug("Traceback for '%s' failure:", name, exc_info=True)
                errors.append(f"{name}: {err}")

        error_summary = "; ".join(errors) if errors else "No eligible providers available"
        raise RuntimeError(f"All TTS providers failed: {error_summary}")

    def synthesize_sync(self, *args, **kwargs) -> str:
        return asyncio.run(self.synthesize(*args, **kwargs))


class VideoManager:
    """Manages video generation and lip-sync providers with fallback chaining."""

    def __init__(self):
        self.providers: dict[str, Any] = {
            "wav2lip": Wav2LipVideoProvider(),
            "ffmpeg": FFmpegVideoProvider(),
        }

    def get_provider_chain(self) -> list[tuple[str, Any]]:
        primary = settings.active_video_provider.lower().strip()
        fallback_order = ["wav2lip", "ffmpeg"]
        chain_keys = [primary] + [k for k in fallback_order if k != primary]

        chain = []
        for key in chain_keys:
            if key in self.providers and (key, self.providers[key]) not in chain:
                chain.append((key, self.providers[key]))
        logger.debug("Resolved Video provider chain: %s", [k for k, _ in chain])
        return chain

    def get_provider(self, name: str | None = None) -> Any:
        key = (name or settings.active_video_provider).lower().strip()
        return self.providers.get(key, self.providers["wav2lip"])

    async def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        **kwargs
    ) -> str:
        errors: list[str] = []
        logger.debug(
            "Video generation requested: video_path=%s, audio_path=%s, output=%s",
            video_path,
            audio_path,
            output_path,
        )

        for name, provider in self.get_provider_chain():
            try:
                logger.info("Attempting video generation with provider: %s", name)
                t0 = time.time()
                result = await provider.generate_video(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    **kwargs
                )
                elapsed = round(time.time() - t0, 3)
                logger.debug("Provider '%s' completed video generation in %ss -> %s", name, elapsed, result)
                return result
            except Exception as err:
                logger.warning("Video provider '%s' failed: %s", name, err)
                logger.debug("Traceback for '%s' video failure:", name, exc_info=True)
                errors.append(f"{name}: {err}")

        error_summary = "; ".join(errors) if errors else "No eligible video providers available"
        raise RuntimeError(f"All video providers failed: {error_summary}")

    def generate_sync(self, *args, **kwargs) -> str:
        return asyncio.run(self.generate(*args, **kwargs))
