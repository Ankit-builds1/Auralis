"""
ML Week 3 Day 4 - Step 2: WavLM embeddings for extra emotion corpora
(RAVDESS and TESS), so the model hears many more speakers than CREMA-D.

Same audio loading (16 kHz, trim, max 6 s) and same embedding (mean over
time of every hidden layer) as extract_ssl_embeddings.py, so all corpora
can be trained together.

Labels are mapped to our 6 emotions:
  RAVDESS: neutral, calm -> neutral, happy, sad, angry, fearful -> fear,
           disgust   (surprised is dropped)
  TESS:    angry, disgust, fear, happy, neutral, sad
           (pleasant surprise 'ps' is dropped)

Run from the ml/ folder:
    python -m src.features.extract_corpus --corpus ravdess --list-only   (check files, no model)
    python -m src.features.extract_corpus --corpus ravdess --model wavlm-large
    python -m src.features.extract_corpus --corpus tess    --model wavlm-large

Input : data/raw/ravdess/**/*.wav   and   data/raw/tess/**/*.wav
Output: data/features/ssl/<model>_<corpus>.npz   (X, y, speaker, files)
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.features.extract_ssl_embeddings import MODELS, OUT_DIR, embed, load_audio


RAW_DIR = Path("data/raw")

RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "neutral",          # calm -> neutral
    "03": "happy", "04": "sad", "05": "angry",
    "06": "fear", "07": "disgust",
    # "08" surprised -> dropped
}

TESS_EMOTIONS = {
    "angry": "angry", "disgust": "disgust", "fear": "fear",
    "happy": "happy", "neutral": "neutral", "sad": "sad",
    # "ps" (pleasant surprise) -> dropped
}


def parse_ravdess(path):
    """03-01-05-01-02-01-12.wav = modality-channel-emotion-intensity-
    statement-repetition-actor. Only speech (channel 01)."""
    parts = path.stem.split("-")
    if len(parts) != 7 or parts[1] != "01":
        return None
    emotion = RAVDESS_EMOTIONS.get(parts[2])
    if emotion is None:
        return None
    return emotion, f"ravdess_{int(parts[6]):02d}"


def parse_tess(path):
    """OAF_back_angry.wav / YAF_youth_ps.wav = speaker_word_emotion."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    emotion = TESS_EMOTIONS.get(parts[-1].lower())
    if emotion is None:
        return None
    speaker = parts[0].upper()
    if speaker.startswith("OA"):       # one file is misnamed "OA_bite_neutral"
        speaker = "OAF"
    elif speaker.startswith("YA"):
        speaker = "YAF"
    return emotion, f"tess_{speaker}"


PARSERS = {"ravdess": parse_ravdess, "tess": parse_tess}


def find_clips(corpus):
    root = RAW_DIR / corpus
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise SystemExit(f"No .wav files found in {root} - download and extract first.")

    clips, seen, skipped = [], set(), Counter()
    for path in files:
        key = path.name.lower()
        if key in seen:                 # TESS zip contains a duplicate copy
            skipped["duplicate"] += 1
            continue
        seen.add(key)

        parsed = PARSERS[corpus](path)
        if parsed is None:
            skipped["not used (surprise / song / unknown name)"] += 1
            continue
        clips.append((path, *parsed))

    return files, clips, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=PARSERS, required=True)
    parser.add_argument("--model", choices=MODELS, default="wavlm-large")
    parser.add_argument("--list-only", action="store_true",
                        help="only count files and labels, do not load the model")
    args = parser.parse_args()

    files, clips, skipped = find_clips(args.corpus)

    print(f"[CORPUS] {args.corpus}: {len(files)} wav files found, {len(clips)} used")
    for reason, n in skipped.items():
        print(f"  skipped {n}: {reason}")
    emotions = Counter(e for _, e, _ in clips)
    speakers = sorted({s for _, _, s in clips})
    print("  emotions: " + ", ".join(f"{e}={n}" for e, n in sorted(emotions.items())))
    print(f"  speakers: {len(speakers)}  ({speakers[0]} ... {speakers[-1]})")

    if args.list_only:
        return

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = MODELS[args.model]
    print(f"[CORPUS] {name} on {device}")
    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    features, labels, spk, names = [], [], [], []
    too_short = 0
    start = time.perf_counter()

    for i, (path, emotion, speaker) in enumerate(clips, start=1):
        audio = load_audio(str(path))
        if len(audio) < 16000 // 4:
            too_short += 1
        else:
            features.append(embed(model, extractor, audio, device).astype(np.float16))
            labels.append(emotion)
            spk.append(speaker)
            names.append(path.name)

        if i % 200 == 0 or i == len(clips):
            rate = (time.perf_counter() - start) / i
            left = rate * (len(clips) - i) / 60
            print(f"  {i}/{len(clips)} clips  (~{left:.1f} min left)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.model}_{args.corpus}.npz"
    np.savez_compressed(out, X=np.stack(features), y=np.array(labels),
                        speaker=np.array(spk), files=np.array(names))
    print(f"[CORPUS] saved {out}  ({len(features)} clips, {too_short} too short)")


if __name__ == "__main__":
    main()