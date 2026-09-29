"""
Outgoing audio track for Auralis.

Anything that produces speech (a test tone today, XTTS/Bark later) calls
push_chunk() with PCM audio, then end_clip() when the clip is complete.
recv() sends the audio to the browser as 20 ms frames, and sends silence
whenever nothing is queued.
"""

import asyncio
import fractions
import threading
import time

import numpy as np
from av import AudioFrame
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError


SAMPLE_RATE = 48000                  # WebRTC / Opus rate
FRAME_SAMPLES = 960                  # 20 ms at 48 kHz
TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class TTSAudioTrack(MediaStreamTrack):

    kind = "audio"

    def __init__(self, on_state=None):
        super().__init__()

        # Optional callback: on_state("started") / on_state("finished").
        # Used from Day 2 to tell the frontend when the AI is speaking.
        self._on_state = on_state

        self._lock = threading.Lock()
        self._buffer = np.zeros(0, dtype=np.int16)

        # True while chunks of the current clip may still be arriving
        self._clip_open = False
        self._speaking = False

        self._start = None
        self._timestamp = 0

        # Stats (used on Day 5)
        self.frames_sent = 0
        self.underruns = 0

    # ------------------------------------------------------
    # PRODUCER SIDE (TTS, test tone, WAV player ...)
    # ------------------------------------------------------

    def push_chunk(self, pcm, sample_rate=SAMPLE_RATE):
        """
        Queue mono PCM audio. Accepts int16, or float in -1..1.
        Safe to call from the event loop or from a worker thread.
        """

        audio = np.asarray(pcm)

        if audio.size == 0:
            return

        if audio.dtype != np.int16:
            audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)

        if sample_rate != SAMPLE_RATE:
            new_length = int(round(len(audio) * SAMPLE_RATE / sample_rate))
            positions = np.linspace(0, len(audio), new_length, endpoint=False)
            audio = np.interp(
                positions,
                np.arange(len(audio)),
                audio,
            ).astype(np.int16)

        with self._lock:
            self._buffer = np.concatenate((self._buffer, audio))
            self._clip_open = True

    def end_clip(self):
        """Call once when no more chunks are coming for this clip."""

        with self._lock:
            self._clip_open = False

    # ------------------------------------------------------
    # CONSUMER SIDE (called by aiortc every 20 ms)
    # ------------------------------------------------------

    def _next_frame_samples(self):

        event = None

        with self._lock:

            available = len(self._buffer)
            samples = np.zeros(FRAME_SAMPLES, dtype=np.int16)
            consumed = False

            if available >= FRAME_SAMPLES:
                samples[:] = self._buffer[:FRAME_SAMPLES]
                self._buffer = self._buffer[FRAME_SAMPLES:]
                consumed = True

            elif available > 0 and not self._clip_open:
                # Final partial piece of a finished clip: pad with silence
                samples[:available] = self._buffer
                self._buffer = np.zeros(0, dtype=np.int16)
                consumed = True

            elif self._speaking and self._clip_open:
                # Audio ran out while the clip is still being produced
                self.underruns += 1

            if consumed and not self._speaking:
                self._speaking = True
                event = "started"

            elif (
                self._speaking
                and len(self._buffer) == 0
                and not self._clip_open
            ):
                self._speaking = False
                event = "finished"

        return samples, event

    async def recv(self):

        if self.readyState != "live":
            raise MediaStreamError

        # Pace output to real time
        if self._start is None:
            self._start = time.time()
            self._timestamp = 0
        else:
            self._timestamp += FRAME_SAMPLES
            wait = (
                self._start
                + self._timestamp / SAMPLE_RATE
                - time.time()
            )
            if wait > 0:
                await asyncio.sleep(wait)

        samples, event = self._next_frame_samples()

        if event and self._on_state:
            try:
                self._on_state(event)
            except Exception as exc:
                print(f"[TTS] on_state error: {type(exc).__name__}: {exc}")

        frame = AudioFrame(
            format="s16",
            layout="mono",
            samples=FRAME_SAMPLES,
        )
        frame.planes[0].update(samples.tobytes())
        frame.pts = self._timestamp
        frame.sample_rate = SAMPLE_RATE
        frame.time_base = TIME_BASE

        self.frames_sent += 1

        return frame


async def play_test_tone(track, seconds=2.0, frequency=440.0, chunk_ms=100):
    """
    Stream a sine tone in small chunks, the way a streaming TTS would.
    Used to prove the audio path works before any real voice exists.
    """

    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
    total_samples = int(SAMPLE_RATE * seconds)

    print(f"[TTS] Test tone started ({seconds} s, {frequency} Hz)")

    for start in range(0, total_samples, chunk_samples):

        count = min(chunk_samples, total_samples - start)

        t = (np.arange(count) + start) / SAMPLE_RATE
        tone = 0.25 * np.sin(2 * np.pi * frequency * t)

        track.push_chunk(tone.astype(np.float32), SAMPLE_RATE)

        # Produce slightly faster than real time so the buffer stays ahead
        await asyncio.sleep(chunk_ms / 1000 * 0.8)

    track.end_clip()

    print("[TTS] Test tone queued completely")