import os
import subprocess
from pathlib import Path
from core.logger import setup_logger

logger = setup_logger("core.extractor")


class MediaExtractor:
    """Extracts reference frames and audio tracks from input media using ffmpeg."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin

    def extract_face_frame(self, video_path: str, output_image_path: str, timestamp_sec: float = 1.0) -> str:
        # Grabs a single clear frame from the video to use as avatar face image
        if not os.path.exists(video_path):
            logger.error("Input video file not found: %s", video_path)
            raise FileNotFoundError(f"Video file not found: {video_path}")

        Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss", str(timestamp_sec),
            "-i", video_path,
            "-vframes", "1",
            "-pix_fmt", "yuv420p",
            "-q:v", "2",
            output_image_path,
        ]

        logger.info("Extracting face frame from %s at offset %ss...", video_path, timestamp_sec)
        logger.debug("FFmpeg frame extraction command: %s", " ".join(cmd))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug("FFmpeg frame extraction returncode=%s", res.returncode)

        # Fall back to frame 0 if the video is shorter than timestamp_sec
        if res.returncode != 0 or not os.path.exists(output_image_path) or os.path.getsize(output_image_path) == 0:
            logger.info("Seeking to %ss failed; extracting first available frame from 0.0s...", timestamp_sec)
            fallback_cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", video_path,
                "-vframes", "1",
                "-pix_fmt", "yuv420p",
                "-q:v", "2",
                output_image_path,
            ]
            logger.debug("FFmpeg fallback frame extraction command: %s", " ".join(fallback_cmd))
            res = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.debug("FFmpeg fallback returncode=%s", res.returncode)
            if res.returncode != 0:
                logger.error("ffmpeg fallback frame extraction failed: %s", res.stderr)
                raise RuntimeError(f"ffmpeg frame extraction failed: {res.stderr}")

        if not os.path.exists(output_image_path) or os.path.getsize(output_image_path) == 0:
            logger.error("Extracted frame missing or empty at %s", output_image_path)
            raise RuntimeError(f"Extracted image is missing or empty at {output_image_path}")

        logger.info("Face frame saved to: %s", output_image_path)
        return output_image_path

    def has_audio_stream(self, media_path: str) -> bool:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            media_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(res.stdout.strip())

    def extract_audio(
        self,
        video_path: str,
        output_audio_path: str,
        max_duration_sec: int = 15,
        isolate_vocals: bool = False,
    ) -> str:
        # Extracts a clean, noise-reduced WAV clip for voice cloning reference.
        # Applies bandpass, FFT noise reduction, and dynamic range normalization.
        if not os.path.exists(video_path):
            logger.error("Input media file not found: %s", video_path)
            raise FileNotFoundError(f"Media file not found: {video_path}")

        if not self.has_audio_stream(video_path):
            raise RuntimeError(f"Input file contains no audio stream: {video_path}")

        Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
        wav_path = str(Path(output_audio_path).with_suffix(".wav"))

        # Audio filter chain:
        # 1. dynaudnorm: normalizes speech volume first so quiet microphone recordings are not lost.
        # 2. highpass/lowpass: vocal bandpass (80Hz to 7.5kHz), cutting bass rumble & high hiss.
        # 3. afftdn: FFT-based background denoiser.
        # 4. adeclick: removes audio click/pop artifacts.
        audio_filters = (
            "dynaudnorm=f=150:g=15,"
            "highpass=f=80,lowpass=f=7500,"
            "afftdn=nf=-25,"
            "adeclick"
        )

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", video_path,
            "-t", str(max_duration_sec),
            "-vn",
            "-af", audio_filters,
            "-acodec", "pcm_s16le",  # Lossless PCM for speech embeddings
            "-ar", "16000",          # Standard 16kHz sample rate for voice models
            "-ac", "1",              # Mono
            wav_path,
        ]

        logger.info(
            "Extracting and cleaning reference audio from %s (max %ss)...",
            video_path,
            max_duration_sec,
        )
        logger.debug("FFmpeg audio extraction command: %s", " ".join(cmd))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug("FFmpeg audio extraction returncode=%s", res.returncode)
        if res.returncode != 0:
            logger.error("ffmpeg audio extraction failed: %s", res.stderr)
            raise RuntimeError(f"ffmpeg audio extraction failed: {res.stderr}")

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            logger.error("Extracted reference audio file is empty: %s", wav_path)
            raise RuntimeError(f"Extracted audio file is missing or empty at {wav_path}")

        if isolate_vocals:
            try:
                wav_path = self._isolate_vocals_demucs(wav_path)
            except Exception as e:
                logger.warning("AI vocal isolation failed: %s. Using filtered track.", e)

        logger.info("Clean reference audio prepared at: %s", wav_path)
        return wav_path

    def _isolate_vocals_demucs(self, audio_wav_path: str) -> str:
        # Strips out background music and sound effects using Demucs AI separation
        import torch
        logger.info("Running Demucs vocal isolation on %s...", audio_wav_path)

        out_dir = Path(audio_wav_path).parent / "separated"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs",
            "--out", str(out_dir),
            audio_wav_path,
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Demucs command failed: {res.stderr}")

        base_name = Path(audio_wav_path).stem
        isolated_vocals_file = out_dir / "htdemucs" / base_name / "vocals.wav"

        if not isolated_vocals_file.exists():
            raise FileNotFoundError(f"Demucs finished but output missing: {isolated_vocals_file}")

        logger.info("Vocals isolated successfully: %s", isolated_vocals_file)
        return str(isolated_vocals_file)
