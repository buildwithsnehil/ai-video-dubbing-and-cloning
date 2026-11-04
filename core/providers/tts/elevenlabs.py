from pathlib import Path
import httpx
from core.logger import setup_logger
from config.config import settings

logger = setup_logger("provider.tts.elevenlabs")


class ElevenLabsProvider:
    """Commercial voice cloning and text-to-speech using ElevenLabs API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.elevenlabs_api_key
        self.default_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (Default)

    async def generate_speech(
        self,
        text: str,
        output_path: str,
        reference_audio_path: str | None = None,
        voice: str | None = None,
        **kwargs
    ) -> str:
        if not self.api_key:
            raise ValueError("elevenlabs_api_key is not set. Please provide a valid API key.")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        voice_id = voice if voice and len(voice) > 10 else self.default_voice_id
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        logger.info("Requesting speech synthesis from ElevenLabs (voice: %s)...", voice_id)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                logger.error("ElevenLabs API failed with status %s: %s", response.status_code, response.text)
                raise RuntimeError(f"ElevenLabs error: {response.text}")

            with open(output_path, "wb") as f:
                f.write(response.content)

        logger.info("ElevenLabs speech written to: %s", output_path)
        return output_path
