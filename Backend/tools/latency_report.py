"""
Week 4 latency sprint: where does the time go?

Reads Backend/logs/latency.csv (one row per turn, written by audio_track.py)
and prints median / p95 for every stage, so a change (GPU Whisper,
VAD_SILENCE_MS, WHISPER_BEAM_SIZE, shorter first sentence ...) can be
measured instead of guessed.

    cd Backend
    python tools/latency_report.py              # last 20 turns
    python tools/latency_report.py --last 10
    python tools/latency_report.py --since 2026-10-24T10:00

All times are measured from the moment the VAD detected end of speech.
The user also waits for the VAD silence window (VAD_SILENCE_MS, default
700 ms) BEFORE that moment.
"""

import argparse
import csv
import math
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BACKEND_DIR / "logs" / "latency.csv"
TARGET_MS = 800


def num(row, key):
    try:
        value = float(row.get(key) or "")
        return None if math.isnan(value) else value
    except ValueError:
        return None


def pct(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    rank = max(0, min(len(values) - 1, math.ceil(p / 100 * len(values)) - 1))
    return values[rank]


def fmt(value):
    return "     -" if value is None else f"{value:6.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=20, help="number of most recent turns")
    parser.add_argument("--since", default=None, help="only turns at/after this ISO time")
    parser.add_argument("--csv", default=str(CSV_PATH))
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"No latency log at {path} - talk to Auralis first (LATENCY_LOG=1).")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.since:
        rows = [r for r in rows if (r.get("time") or "") >= args.since]
    rows = rows[-args.last:]
    if not rows:
        raise SystemExit("No turns match.")

    stages = []
    for r in rows:
        stt, ttft = num(r, "stt_ms"), num(r, "ttft_ms")
        piece, ttfa = num(r, "first_piece_ms"), num(r, "ttfa_ms")
        synth = num(r, "first_synth_ms")
        stages.append({
            "speech -> text (Whisper)": stt,
            "text -> first word (Llama)": (ttft - stt) if None not in (ttft, stt) else None,
            "first word -> first phrase": (piece - ttft) if None not in (piece, ttft) else None,
            "first phrase -> audio (Piper)": (ttfa - piece) if None not in (ttfa, piece) else None,
            "   of which pure synthesis": synth,
            "FIRST AUDIO (TTFA)": ttfa,
            "full reply text": num(r, "total_ms"),
        })

    print(f"\nAuralis latency report: {len(rows)} turns "
          f"({rows[0].get('time', '?')} -> {rows[-1].get('time', '?')})")
    env = {}
    try:
        from dotenv import dotenv_values
        env = dotenv_values(BACKEND_DIR / ".env")
    except Exception:
        pass
    setting = lambda k, d: env.get(k) or os.getenv(k) or d
    print(f"current .env: WHISPER_BEAM_SIZE={setting('WHISPER_BEAM_SIZE', '5')}, "
          f"VAD_SILENCE_MS={setting('VAD_SILENCE_MS', '700')}, "
          f"BARGE_IN_GATE_DB={setting('BARGE_IN_GATE_DB', '5')}\n")
    print(f"{'stage (ms, from end of speech)':32s} {'median':>6s} {'p95':>6s} {'mean':>6s}")
    print("-" * 56)
    for key in stages[0]:
        values = [s[key] for s in stages if s[key] is not None]
        mean = sum(values) / len(values) if values else None
        print(f"{key:32s} {fmt(pct(values, 50))} {fmt(pct(values, 95))} {fmt(mean)}")

    ttfa = [s["FIRST AUDIO (TTFA)"] for s in stages if s["FIRST AUDIO (TTFA)"] is not None]
    if ttfa:
        hit = sum(v <= TARGET_MS for v in ttfa) / len(ttfa)
        print(f"\nturns with first audio <= {TARGET_MS} ms: {hit:.0%}")
        biggest = max(
            (k for k in stages[0] if k not in ("FIRST AUDIO (TTFA)", "full reply text",
                                               "   of which pure synthesis")),
            key=lambda k: pct([s[k] for s in stages], 50) or 0,
        )
        print(f"slowest stage (median): {biggest.strip()}")


if __name__ == "__main__":
    main()
