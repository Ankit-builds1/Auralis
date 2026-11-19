import logging
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

# Make the ML team's code importable: Backend/ml must be on the Python path.
# parents[2] goes from ml_clients -> app -> Backend. Change "ml" if your folder is named differently.
ML_ROOT = Path(__file__).resolve().parents[2] / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from app.ml_clients.emotion.audio_emotion import predict_emotion_from_audio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_EMOTION = "neutral"


def predict_emotion_from_waveform(audio: np.ndarray) -> str:
    """
    Convert a 16 kHz mono float32 waveform into a temporary WAV file
    and use the ML team's emotion model to predict the emotion.

    Blocking call: run it with asyncio.to_thread(...) from async code.
    Returns "neutral" if the audio is empty or the model fails.
    """

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size == 0:
        return DEFAULT_EMOTION

    # Convert float32 [-1, 1] audio to 16-bit PCM.
    pcm_audio = np.clip(audio, -1.0, 1.0)
    pcm_audio = (pcm_audio * 32767).astype(np.int16)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm_audio.tobytes())

        emotion = predict_emotion_from_audio(temp_path)
        return str(emotion)

    except Exception:
        # Don't let an emotion failure stop the LLM response.
        logger.exception("Emotion prediction failed, using default")
        return DEFAULT_EMOTION

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)