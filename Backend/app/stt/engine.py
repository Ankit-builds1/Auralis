"""
One shared Whisper model for the whole server.

Loaded and warmed up ONCE at server start (see main.py), instead of on
the first sentence. Every connection then reuses the same model.
"""

import threading

from app.stt.transcriber import SpeechToText


_stt = None
_lock = threading.Lock()


def get_stt():
    """Return the shared SpeechToText instance (creates it if needed)."""
    global _stt

    if _stt is None:
        with _lock:
            if _stt is None:
                _stt = SpeechToText()

    return _stt


def load_and_warmup():
    """Blocking. Call from a thread at server start."""
    stt = get_stt()
    stt.warmup()
    return stt