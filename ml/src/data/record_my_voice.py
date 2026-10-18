"""
ML Week 3 Day 3 - Step 1: record our own emotion clips (speaker adaptation).

Why: the live test showed the CREMA-D model is often confidently wrong on
our voices (Indian-English accent, laptop mic, natural instead of acted
speech). We record a small labelled set of our own voices and add it to
training.

How it works (CREMA-D style): the SAME 10 neutral sentences are spoken in
all 6 emotions, so the model has to learn the TONE, not the words.
10 sentences x 6 emotions = 60 clips (~15-20 minutes).

Run from the ml/ folder (one person at a time, on their own laptop):
    python -m src.data.record_my_voice --speaker ankit
    python -m src.data.record_my_voice --speaker priya

Clips are saved to:
    data/my_voice/<speaker>/<emotion>/<speaker>_<emotion>_<NN>.wav
Already-recorded clips are skipped, so you can stop (Ctrl+C) and continue
later. These are personal voice recordings: they are NOT pushed to git.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
SECONDS = 4.0
OUT_DIR = Path("data/my_voice")

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]

# Neutral words on purpose: the emotion must come from the voice.
SENTENCES = [
    "I am going to the shop now.",
    "The meeting is at five o'clock.",
    "I just got the message from him.",
    "We have to finish this today.",
    "That is what I was talking about.",
    "Someone is standing at the door.",
    "I did not expect this at all.",
    "It is already very late now.",
    "Tell me what happened yesterday.",
    "I think we should go home.",
]

HOW_TO_SAY = {
    "neutral": "flat, normal, like reading a timetable",
    "happy":   "bright, smiling, a little faster and higher",
    "sad":     "slow, low, quiet, tired",
    "angry":   "loud, sharp, hard on the words",
    "fear":    "tense, shaky, a bit fast and high, worried",
    "disgust": "slow, 'ugh' tone, pulled back, unpleasant",
}

QUIET_PEAK = 0.05      # below this the clip is probably too quiet
CLIP_PEAK = 0.99       # above this the clip is clipped (too loud)


def record_clip():
    for n in (3, 2, 1):
        print(f"   {n}...", end=" ", flush=True)
        time.sleep(0.6)
    print("SPEAK NOW", flush=True)

    audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def check_level(audio):
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak < QUIET_PEAK:
        return peak, "TOO QUIET - speak louder / closer to the mic"
    if peak > CLIP_PEAK:
        return peak, "TOO LOUD (clipped) - move back a little"
    return peak, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True, help="your name, e.g. ankit")
    parser.add_argument("--emotions", nargs="*", default=EMOTIONS,
                        help="record only some emotions, e.g. --emotions sad fear")
    args = parser.parse_args()

    speaker = args.speaker.strip().lower()
    todo = []
    for emotion in args.emotions:
        for i, sentence in enumerate(SENTENCES, start=1):
            path = OUT_DIR / speaker / emotion / f"{speaker}_{emotion}_{i:02d}.wav"
            if not path.exists():
                todo.append((emotion, i, sentence, path))

    total = len(args.emotions) * len(SENTENCES)
    print(f"\n[REC] speaker '{speaker}': {total - len(todo)}/{total} clips already done, "
          f"{len(todo)} to record")
    print("[REC] Keys after each clip:  Enter = keep   r = redo   q = quit\n")

    current = None
    for emotion, i, sentence, path in todo:
        if emotion != current:
            current = emotion
            print("=" * 60)
            print(f"  EMOTION: {emotion.upper()}  ->  {HOW_TO_SAY[emotion]}")
            print("=" * 60)

        while True:
            print(f"\n[{emotion} {i:02d}/{len(SENTENCES)}]  say:  \"{sentence}\"")
            input("   press Enter when ready ")
            audio = record_clip()
            peak, verdict = check_level(audio)
            print(f"   peak {peak:.2f} -> {verdict}")

            choice = input("   Enter = keep, r = redo, q = quit: ").strip().lower()
            if choice == "q":
                print("[REC] Stopped. Run the same command again to continue.")
                return
            if choice == "r":
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, SAMPLE_RATE)
            print(f"   saved {path}")
            break

    print(f"\n[REC] Done: all {total} clips for '{speaker}' are in {OUT_DIR / speaker}")


if __name__ == "__main__":
    main()
