"""
ML Day 2 - Task 2: emotion -> voice style for the Piper TTS.

Piper has no emotion control, so each style changes:
    length_scale  > 1.0 = slower speech        (Piper setting)
    noise_scale   lower = flatter, calmer tone (Piper setting)
    noise_w_scale lower = steadier rhythm      (Piper setting)
    volume        0..1                         (Piper setting)
    pitch_steps   semitones, applied after Piper with librosa
"""

import numpy as np

GROUPS = {
    "angry": "upset", "disgust": "upset", "fear": "upset",
    "sad": "low",
    "happy": "okay", "neutral": "okay",
}

MIN_CONFIDENCE = 0.45

STYLES = {
    "upset": {
        "length_scale": 1.3,
        "noise_scale": 0.4,
        "noise_w_scale": 0.5,
        "volume": 0.85,
        "pitch_steps": -1.0,
        "llm_hint": "The user sounds upset or stressed. Stay calm and reassuring. "
                    "Use short, slow, simple sentences.",
    },
    "low": {
        "length_scale": 1.4,
        "noise_scale": 0.4,
        "noise_w_scale": 0.5,
        "volume": 0.7,
        "pitch_steps": -2.0,
        "llm_hint": "The user sounds sad or low. Be gentle, warm and supportive.",
    },
    "okay": {
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w_scale": 0.8,
        "volume": 1.0,
        "pitch_steps": 0.0,
        "llm_hint": "The user sounds fine. Be friendly and natural.",
    },
}


def emotion_group(label, confidence=None):
    if confidence is not None and confidence < MIN_CONFIDENCE:
        return "okay"
    return GROUPS.get(label, "okay")


def voice_style(label, confidence=None):
    group = emotion_group(label, confidence)
    return group, STYLES[group]


def piper_config(style):
    from piper import SynthesisConfig

    return SynthesisConfig(
        length_scale=style["length_scale"],
        noise_scale=style["noise_scale"],
        noise_w_scale=style["noise_w_scale"],
        volume=style["volume"],
    )


def apply_voice_effects(audio, sample_rate, style):
    """Pitch shift after Piper (no-op for the neutral style)."""
    steps = style.get("pitch_steps", 0.0)
    if not steps:
        return audio

    import librosa

    shifted = librosa.effects.pitch_shift(
        np.asarray(audio, dtype=np.float32), sr=sample_rate, n_steps=steps
    )
    return np.clip(shifted, -1.0, 1.0).astype(np.float32)
