import asyncio
from pathlib import Path
from wav2lip import Wav2LipPipeline
from config.config import settings
from core.logger import setup_logger

logger = setup_logger("provider.video.wav2lip")


class Wav2LipVideoProvider:
    """Local lip-sync video generator utilizing packaged in-memory Wav2LipPipeline."""

    def __init__(
        self,
        checkpoint_path: str = "gan",
    ):
        self.checkpoint_path = checkpoint_path

        # CPU performance: resize_factor=2 yields ~3-4x faster inference on CPU
        self.resize_factor = int(settings.wav2lip_resize_factor)

        # nosmooth=False enables temporal frame blending for smoother lip transitions
        self.nosmooth = settings.wav2lip_nosmooth

        # Face crop padding: top bottom left right (default 0 10 0 0 expands crop downward for chin coverage)
        self.pads = settings.wav2lip_pads
        self._pipeline: Wav2LipPipeline | None = None

    async def generate_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        **kwargs,
    ) -> str:
        input_video = video_path or kwargs.get("image_path")
        resolved_video = Path(input_video).resolve()
        resolved_audio = Path(audio_path).resolve()
        resolved_output = Path(output_path).resolve()

        if not resolved_video.exists():
            raise FileNotFoundError(f"Source video not found: {resolved_video}")
        if not resolved_audio.exists():
            raise FileNotFoundError(f"Synthesized audio source not found: {resolved_audio}")

        logger.info(
            "Starting Wav2Lip lip-sync (video=%s, audio=%s, resize=%s, nosmooth=%s, pads=%s)",
            resolved_video.name,
            resolved_audio.name,
            self.resize_factor,
            self.nosmooth,
            self.pads,
        )

        loop = asyncio.get_running_loop()
        if self._pipeline is None:
            self._pipeline = Wav2LipPipeline(checkpoint=self.checkpoint_path)

        pads_tuple = tuple(map(int, self.pads.split()))
        result_path = await loop.run_in_executor(
            None,
            lambda: self._pipeline.generate(
                face=resolved_video,
                audio_path=resolved_audio,
                outfile=resolved_output,
                resize_factor=self.resize_factor,
                pads=pads_tuple,
                nosmooth=self.nosmooth,
            ),
        )

        logger.info("Wav2Lip video generated successfully: %s", resolved_output.name)
        return str(result_path)
