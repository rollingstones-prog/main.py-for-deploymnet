# ElevenLabs API integration for TTS and STT
# Add your ElevenLabs API key to .env as ELEVENLABS_API_KEY
import requests
import os
import tempfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Default voice_id (set in .env)
ELEVENLABS_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")


# ---------------------------
# Helper: Convert with ffmpeg
# ---------------------------
def convert_to_mp3(input_path: Path) -> Path:
    """
    Convert input audio (ogg/opus/wav) to mp3 using ffmpeg CLI.
    Returns path to converted mp3.
    """
    out_path = Path(tempfile.gettempdir()) / (input_path.stem + ".mp3")
    try:
        cmd = ["ffmpeg", "-y", "-i", str(input_path), str(out_path)]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return out_path
    except Exception as e:
        print("❌ ffmpeg conversion failed:", str(e))
        return input_path  # fallback: return original


# ---------------------------
# TTS: Text → Speech (MP3)
# ---------------------------
def tts_to_mp3(text, out_path, voice_id=ELEVENLABS_VOICE_ID):
    """
    Convert text to speech using ElevenLabs API and save as MP3.
    """
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return out_path
        else:
            print("❌ ElevenLabs TTS error:", resp.status_code, resp.text)
            return None
    except Exception as e:
        print("❌ ElevenLabs TTS exception:", str(e))
        return None


# ---------------------------
# STT: Speech → Text
# ---------------------------\
def stt_from_mp3(audio_path: Path):
    url = ELEVENLABS_STT_URL
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    try:
        audio_path = Path(audio_path)

        # Convert if OGG/OPUS/WAV
        if audio_path.suffix.lower() in [".ogg", ".opus", ".wav"]:
            use_path = convert_to_mp3(audio_path)
        else:
            use_path = audio_path

        with open(use_path, "rb") as f:
            files = {"file": f}  # 👈 API ko "file" chahiye, "audio" nahi
            data = {"model_id": "scribe_v1"}
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)

        if resp.status_code == 200:
            result = resp.json()
            return result.get("text", "").strip()
        else:
            print("❌ ElevenLabs STT error:", resp.status_code, resp.text)
            return "Audio transcription me temporary issue hai."

    except Exception as e:
        print("❌ ElevenLabs STT exception:", str(e))
        return "Audio transcription me temporary issue hai."
