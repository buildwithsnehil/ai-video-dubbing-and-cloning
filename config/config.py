from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # Audio synthesis and voice cloning
    active_tts_provider: str = Field(default="jugnu", description="TTS engine: 'jugnu', 'edge_tts', or 'elevenlabs'")
    jugnu_model_variant: str = Field(default="base", description="Jugnu model variant ('base', 'lite')")
    jugnu_temperature: float = Field(default=0.3, description="ODE sampling temperature for voice stability")
    jugnu_max_tokens: int = Field(default=75, description="Maximum tokens per generation chunk")
    elevenlabs_api_key: str = Field(default=None, description="ElevenLabs API key for cloud voice cloning")
    hf_token: str = Field(default="", description="Hugging Face token for gated model access")

    # Video generation and lip-sync
    active_video_provider: str = Field(default="wav2lip", description="Video engine: 'wav2lip' or 'ffmpeg'")
    wav2lip_resize_factor: int = Field(default=1, description="Downscale factor for Wav2Lip inference (1=full quality, 2=faster CPU)")
    wav2lip_nosmooth: bool = Field(default=False, description="Disable temporal smoothing in Wav2Lip (False=smoother lips, True=faster)")
    wav2lip_pads: str = Field(default="0 10 0 0", description="Face crop padding in pixels: top bottom left right")

    # Storage Directories
    upload_dir: str = Field(default="uploads", description="Directory for uploaded reference audio/video")
    output_dir: str = Field(default="outputs", description="Directory for generated audio outputs")
    temp_dir: str = Field(default="temp", description="Directory for intermediate processing files")
    template_dir: str = Field(default="templates", description="Directory for HTML templates")

    # Logging
    log_level: str = Field(default="INFO", description="Logging verbosity: 'DEBUG', 'INFO', 'WARN', 'ERROR'")
    log_file: str = Field(default="logs/api.log", description="Path to application log file")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
