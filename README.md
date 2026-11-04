# AvatarSync: AI Voice Cloning & Talking Avatar Video Generator

AvatarSync is a modular FastAPI and CLI platform for zero-shot voice cloning, studio vocal mastering, and synchronized talking avatar video generation. It brings together local neural speech models and GAN-based facial lip-syncing with robust cloud fallbacks.

---

## Key Features

- **Local AI Voice Cloning**: AltSlate Jugnu Pocket TTS (`altslate/jugnu-pocket-tts` base and lite models) running locally on CPU or GPU with zero cloud API costs.
- **Automated Studio Vocal Mastering**: Built-in FFmpeg mastering chain (75 Hz highpass rumble filter, 3.5 kHz presence EQ boost for consonant clarity, dynamic compression, and -16 LUFS broadcast loudnorm).
- **Intelligent Sentence Segmentation**: Automatic Devanagari punctuation normalization (`।`, `॥` to `.`) and clause chunking (10–18 words / 35–65 tokens) to eliminate dropped words and tone drift.
- **Talking Avatar Video Lip-Sync**: Local Wav2Lip GAN model running on CPU/GPU for realistic lip synchronization with avatar face frames.
- **Modular Provider Architecture with Fallback Chaining**:
  - **TTS Engine** (`TTSManager`): AltSlate Jugnu Pocket TTS &rarr; Microsoft Edge Neural TTS &rarr; ElevenLabs API.
  - **Video Engine** (`VideoManager`): Wav2Lip GAN &rarr; D-ID Talks API &rarr; FFmpeg static-frame preview generator.
- **Reference Media Extraction**: Automatic face frame extraction (`MediaExtractor.extract_face_frame`) and clean reference audio extraction (`MediaExtractor.extract_audio`) with optional Meta Demucs (`htdemucs`) deep vocal isolation.
- **Secure Media Management**: Streaming and deletion endpoints with strict path traversal validation for both uploaded inputs and generated outputs.
- **Developer Interfaces**: Interactive Swagger API docs (`/docs`), Web Studio UI (`/`), and REST API endpoints (`/api/v1`).

---

## Architecture & Codebase Layout

```
avatarsync-video-generator/
├── app.py                         # FastAPI ASGI entrypoint & server runner
├── api/                           # API routing layer
│   ├── deps.py                    # Shared dependency injection (AvatarService)
│   └── v1/
│       ├── router.py              # Aggregated v1 API router (/api/v1)
│       └── endpoints/             # API endpoint controllers
│           ├── avatar.py          # Avatar generation (/generate) & job status (/jobs/{id})
│           ├── health.py          # Health check endpoint (/health)
│           └── media.py           # Secure media streaming & deletion endpoints
├── services/                      # Application service orchestration
│   └── avatar_service.py          # Coordinates extraction, speech synthesis, lip-sync & jobs
├── config/
│   ├── __init__.py                # Settings export
│   └── config.py                  # Pydantic application settings & .env parser
├── core/
│   ├── extractor.py               # MediaExtractor: face extraction, audio filter chain, Demucs
│   ├── logger.py                  # Application logging configuration (console & file)
│   ├── manager.py                 # TTSManager & VideoManager provider fallback chains
│   ├── models.py                  # Pydantic schemas (HealthResponse, JobData, APIResponse)
│   └── providers/
│       ├── tts/                   # Speech providers: Jugnu (local), Edge TTS, ElevenLabs
│       │   ├── edge.py
│       │   ├── elevenlabs.py
│       │   └── jugnu.py
│       └── video/                 # Video providers: Wav2Lip GAN, FFmpeg fallback
│           ├── ffmpeg.py
│           └── wav2lip_video.py
├── templates/
│   └── index.html                 # Web Studio user interface
├── uploads/                       # Storage for uploaded reference audio/video files
├── outputs/                       # Storage for generated speech audio and lip-synced videos
├── temp/                          # Scratch space for intermediate processing files
├── logs/                          # Application log files (api.log)
├── requirements.txt               # Python package dependencies
├── .env.example                   # Environment variable template
└── README.md                      # Project documentation
```

---

## Prerequisites

Before starting, ensure you have the following installed on your system:

- **Python 3.10+**: Verify with `python3 --version`
- **Git**: Verify with `git --version`
- **FFmpeg** (required for vocal mastering, audio conversion, and video frames):
  - **Ubuntu / Debian:** `sudo apt update && sudo apt install -y ffmpeg`
  - **macOS:** `brew install ffmpeg`
  - **Windows:** `winget install Gyan.FFmpeg` or `choco install ffmpeg`

---

## Setup & Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/<username>/avatarsync-video-generator.git
cd avatarsync-video-generator
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```
*(On Windows: `venv\Scripts\activate`)*

### 3. Install Dependencies

Upgrade pip and install all required Python libraries:

```bash
pip install --upgrade pip

# Optional for CPU-only systems (installs lightweight PyTorch without 4GB CUDA bloat):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies:
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your `.env` configuration file from the template:

```bash
cp .env.example .env
```

The default configuration is **100% free and open-source**:
```ini
# Voice synthesis: 'jugnu' (Free Local AI) or 'edge_tts' (Free Cloud Neural)
ACTIVE_TTS_PROVIDER=jugnu
JUGNU_MODEL_VARIANT=base
JUGNU_TEMPERATURE=0.3
JUGNU_MAX_TOKENS=75

# Video lip-sync: 'wav2lip' (Free Local GAN) or 'ffmpeg' (Preview)
ACTIVE_VIDEO_PROVIDER=wav2lip
WAV2LIP_RESIZE_FACTOR=2       # 2=fast CPU inference, 1=full resolution GPU inference
WAV2LIP_NOSMOOTH=false        # false=smooth temporal lip transitions
WAV2LIP_PADS=0 10 0 0         # Chin padding: top bottom left right

# Storage Directories
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
TEMP_DIR=temp

# Optional commercial API keys (not required for free local operation)
HF_TOKEN=
ELEVENLABS_API_KEY=
```

### 5. Verify Installation

Run a quick import test to verify all providers are loaded:

```bash
python -c "from core.manager import TTSManager, VideoManager; print('AvatarSync is ready!')"
```

---

## Running the Application

### Start FastAPI Server

```bash
source venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8003 --reload
```

- **Interactive Swagger UI:** [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs)
- **Web Studio UI:** [http://127.0.0.1:8003/](http://127.0.0.1:8003/)

---

## REST API Specification (`/api/v1`)

### 1. Health Check
* **Method:** `GET /api/v1/health`
* **Description:** Check operational status and service health.

```bash
curl http://127.0.0.1:8003/api/v1/health
```

---

### 2. Generate Cloned Speech & Talking Avatar Video
* **Method:** `POST /api/v1/avatar/generate`
* **Content-Type:** `multipart/form-data`

#### Request Parameters:
* **`script`** *(text, required)*: Text dialogue to synthesize (supports English, Hindi, and Hinglish).
* **`video`** *(file, optional)*: Reference speaker video/audio (`.mp4`, `.webm`, `.wav`, `.mp3`). If omitted, natural preset voices are used.
* **`voice`** *(text, optional)*: Preset voice ID (e.g., `hi-IN-MadhurNeural`, `en-IN-PrabhatNeural`).
* **`isolate_vocals`** *(boolean, optional, default: `false`)*: AI vocal isolation (Demucs) to strip background noise from input media.
* **`enhance`** *(boolean, optional, default: `true`)*: Apply studio presence EQ, compression, and loudness mastering.
* **`generate_video`** *(boolean, optional, default: `true`)*: Generate synchronized Wav2Lip lip-sync avatar video.

#### Example Request:
```bash
curl -X POST http://127.0.0.1:8003/api/v1/avatar/generate \
  -F "video=@reference_speaker.mp4;type=video/mp4" \
  -F "script=नमस्ते! मैं अवतारसिंक स्टूडियो हूँ।" \
  -F "enhance=true" \
  -F "generate_video=true"
```

#### Example Response (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "job_id": "c4e1f4f6",
    "status": "completed",
    "audio_url": "/api/v1/avatar/c4e1f4f6_audio.mp3",
    "video_url": "/api/v1/avatar/c4e1f4f6_video.mp4",
    "processing_time_sec": 14.2,
    "created_at": "2026-11-03T08:30:15.123456+00:00"
  },
  "message": "Voice synthesized and avatar video generated."
}
```

---

### 3. Check Job Status
* **Method:** `GET /api/v1/avatar/jobs/{job_id}`
* **Description:** Retrieve processing status and URLs for a previously submitted job.

```bash
curl http://127.0.0.1:8003/api/v1/avatar/jobs/c4e1f4f6
```

---

### 4. Stream or Download Media
Stream or download generated videos/audio or uploaded source files.

* **Primary Route:** `GET /api/v1/avatar/{type}/{filename}`
  * `type`: Either `generated` (for outputs) or `uploaded` (for user inputs).
* **Shorthand Route:** `GET /api/v1/avatar/{filename}` (defaults to `generated` for backward compatibility).

#### Example:
```bash
# Stream generated video
curl http://127.0.0.1:8003/api/v1/avatar/generated/c4e1f4f6_video.mp4 --output avatar.mp4

# Or via shorthand URL
curl http://127.0.0.1:8003/api/v1/avatar/c4e1f4f6_video.mp4 --output avatar.mp4
```

---

### 5. Remove Media
Securely delete uploaded media files or generated audio/video artifacts.

* **Primary Route:** `DELETE /api/v1/avatar/{type}/{filename}`
  * `type`: Either `generated` or `uploaded`.
* **Shorthand Route:** `DELETE /api/v1/avatar/{filename}` (searches and removes from both directories).

#### Example Request:
```bash
curl -X DELETE http://127.0.0.1:8003/api/v1/avatar/generated/c4e1f4f6_video.mp4
```

#### Example Response (`200 OK`):
```json
{
  "status": "success",
  "data": null,
  "message": "Media file 'c4e1f4f6_video.mp4' deleted successfully from generated."
}
```

## License

This project is licensed under the MIT License.
