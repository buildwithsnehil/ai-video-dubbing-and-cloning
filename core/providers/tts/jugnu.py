import os
import re
import subprocess
import wave
from pathlib import Path
import numpy as np
from core.logger import setup_logger
from config.config import settings

logger = setup_logger("provider.tts.jugnu")


def _sanitize_script(text: str, target_words: int = 14) -> str:
    """Pre-processes text to support unlimited length without missing words or tone jumps.

    1. Maps Devanagari full stops (। and ॥) to standard periods.
    2. Segments long run-on sentences (>18 words) into balanced, complete sentences
       at natural phrase/clause boundaries so each chunk stays in the optimal 35-65 token range.
    3. Normalizes punctuation to prevent acoustic model confusion.
    """
    text = text.replace("\u0964", ". ").replace("\u0965", ". ")
    text = re.sub(r"\s+", " ", text).strip()

    raw_sentences = [s.strip() for s in re.split(r"([.!?\n]+)", text) if s.strip()]
    output_sentences = []

    i = 0
    while i < len(raw_sentences):
        sent = raw_sentences[i]
        if re.match(r"^[.!?\n]+$", sent):
            i += 1
            continue

        punct = "."
        if i + 1 < len(raw_sentences) and re.match(r"^[.!?\n]+$", raw_sentences[i + 1]):
            punct = raw_sentences[i + 1][0]
            i += 1

        words = sent.split()
        if len(words) <= target_words + 4:
            output_sentences.append(sent + punct)
        else:
            clauses = [c.strip() for c in re.split(r"[,;:—–]+", sent) if c.strip()]
            cur_clause = []
            cur_words = 0
            for cl in clauses:
                cl_len = len(cl.split())
                if cur_words >= 6 and (cur_words + cl_len > target_words):
                    output_sentences.append(", ".join(cur_clause) + ".")
                    cur_clause = [cl]
                    cur_words = cl_len
                else:
                    cur_clause.append(cl)
                    cur_words += cl_len

            if cur_clause:
                last_text = ", ".join(cur_clause)
                if len(last_text.split()) < 4 and output_sentences:
                    output_sentences[-1] = output_sentences[-1].rstrip(".") + ", " + last_text + punct
                else:
                    output_sentences.append(last_text + punct)
        i += 1

    cleaned = []
    for s in output_sentences:
        s = re.sub(r"[,;:—–]+\s*\.", ".", s)
        s = re.sub(r",+", ",", s)
        s = re.sub(r"\.+", ".", s)
        s = re.sub(r"\s+", " ", s)
        cleaned.append(s.strip())

    return " ".join(cleaned)


# Professional vocal mastering filter chain:
# 1. highpass: cuts sub-bass rumble (<75 Hz) for clean low-end.
# 2. equalizer @ 300Hz: dips muddy/boxy cardboard frequencies.
# 3. equalizer @ 3500Hz: presence boost (+3.5 dB) for crisp consonant clarity (T, S, P, K).
# 4. equalizer @ 6000Hz: subtle air and brightness (+2.0 dB).
# 5. compand: gentle dynamic compressor to bring the voice forward and full.
# 6. loudnorm: broadcast-standard loudness (-16 LUFS, -1.5 dB true peak).
VOCAL_MASTERING_FILTERS = (
    "highpass=f=75,"
    "equalizer=f=300:width_type=o:w=1.0:g=-2.0,"
    "equalizer=f=3500:width_type=o:w=1.2:g=3.5,"
    "equalizer=f=6000:width_type=o:w=1.0:g=2.0,"
    "compand=attacks=0.02:decays=0.2:points=-60/-60|-24/-18|-12/-10|0/-3:gain=1.5,"
    "loudnorm=I=-16:TP=-1.5:LRA=10"
)


class JugnuTTSProvider:
    """Local, open-source voice cloning & TTS provider using AltSlate Jugnu Pocket TTS."""

    def __init__(self, model_variant: str | None = None, temperature: float | None = None):
        self.model_variant = model_variant or settings.jugnu_model_variant
        self.temperature = temperature if temperature is not None else float(settings.jugnu_temperature)
        self._model = None

    def _get_model(self):
        # Lazy load the model on first request
        if self._model is None:
            hf_token = settings.hf_token
            if hf_token:
                from huggingface_hub import login
                login(token=hf_token)

            logger.info("Loading Jugnu Pocket TTS (%s model, temp=%.2f)...", self.model_variant, self.temperature)
            from jugnu_tts import Jugnu
            self._model = Jugnu(self.model_variant, temperature=self.temperature)
            logger.info("Jugnu Pocket TTS model loaded.")
        return self._model

    async def generate_speech(
        self,
        text: str,
        output_path: str,
        reference_audio_path: str | None = None,
        voice: str | None = None,
        **kwargs
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tts = self._get_model()

        if not reference_audio_path or not os.path.exists(reference_audio_path):
            raise ValueError("Reference audio sample is required for Jugnu voice cloning.")

        logger.info("Cloning voice embedding from reference audio (%s)...", reference_audio_path)
        cloned_voice = tts.clone(reference_audio_path)

        clean_text = _sanitize_script(text)
        logger.debug("Sanitized script: '%s' -> '%s'", text, clean_text)
        max_tokens = int(settings.jugnu_max_tokens)
        logger.info("Synthesizing speech via Jugnu TTS (max_tokens=%d)...", max_tokens)
        raw_wav = str(Path(output_path).with_name(f"raw_{Path(output_path).stem}.wav"))

        # Generate audio directly with tuned max_tokens to prevent dropped words
        audio_tensor = tts._tts.generate_audio(cloned_voice.state, clean_text, max_tokens=max_tokens)
        pcm = (np.clip(audio_tensor.cpu().numpy(), -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(raw_wav, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(tts.sample_rate)
            f.writeframes(pcm.tobytes())

        if not os.path.exists(raw_wav) or os.path.getsize(raw_wav) == 0:
            raise RuntimeError("Jugnu speak() produced no output audio.")

        # Apply studio vocal mastering chain (enabled by default)
        enhance = kwargs.get("enhance", True)
        af_chain = VOCAL_MASTERING_FILTERS if enhance else "anull"

        if output_path.endswith(".mp3"):
            cmd = [
                "ffmpeg", "-y", "-i", raw_wav,
                "-af", af_chain,
                "-acodec", "libmp3lame", "-q:a", "0",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", raw_wav,
                "-af", af_chain,
                "-acodec", "pcm_s16le", "-ar", "24000",
                output_path,
            ]

        logger.info("Applying audio processing (enhance=%s)...", enhance)
        logger.debug("FFmpeg vocal mastering command: %s", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug("FFmpeg vocal mastering returncode=%s", result.returncode)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio processing failed: {result.stderr}")

        if os.path.exists(raw_wav):
            try:
                os.remove(raw_wav)
            except OSError:
                pass

        logger.info("Speech audio written to: %s (%d bytes)", output_path, os.path.getsize(output_path))
        return output_path
