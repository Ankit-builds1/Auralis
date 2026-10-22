"""
ML Week 4 Day 1 - barge-in threshold evaluation (BARGE_IN_MIN_VOICED_FRAMES).

Question: how many voiced 20 ms frames must a speech segment contain, while
Auralis is talking, before we treat it as the REAL user (barge-in) and not as
leftover ECHO of Auralis's own voice?

What this script does
  1. REAL user speech : my recorded clips  data/my_voice/<speaker>/<emotion>/*.wav
  2. ECHO             : Auralis replies synthesised with the same Piper voice and
                        styles as Backend/app/tts/engine.py, resampled to 16 kHz.
                        Level calibration: each reply is first scaled to the same
                        loudness as the user's own speech (= Auralis as loud at the
                        mic as the user, i.e. echo cancellation OFF). Leftover echo
                        after browser echo cancellation is then simulated by
                        lowering it by -15 / -20 / -25 dB (plus -10 dB as an extra
                        stress case), adding a 20-40 ms room reflection and light
                        noise.
  3. VAD simulation   : same as Backend/app/audio/vad/detector.py
                        (Silero ONNX, 512-sample chunks, threshold 0.5,
                        3 chunks to start a segment, 22 chunks to end it),
                        fed in 20 ms frames. A frame is "voiced" when the latest
                        VAD probability >= threshold (as in processor.py).
  4. Counting rule (ASSUMPTION - barge-in code was not on any remote branch):
                        voiced frames are counted from the START of a segment
                        (the segment opens when the VAD says "in speech").
                        Barge-in fires when the count reaches N.
  5. Two sweeps over N = 10..40 frames:
       a) frame count only (what the backend does now)
       b) frame count + LOUDNESS GATE: a voiced frame only counts when its
          level is >= the gate (dBFS). Finding from the first sweep: Silero
          keeps calling the echo "speech", so the count alone cannot separate
          echo from the user; the echo is simply quieter.
  6. Recommendation: smallest N (then most robust gate) with false triggers
     <= 5% at -20 dB and detection >= 95%. If none: says so.

Limitations: one speaker; real clips are played alone (user speech mixed
on top of live echo is not simulated); echo levels are relative to the user's
speech level, so an absolute dBFS gate would need calibration on the backend.

Run from the ml folder:   python -m src.bargein.eval_barge_in
"""

import os

os.environ.setdefault("HF_HOME", r"D:\Auralis-ML-Work\hf-cache")

import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from piper import PiperVoice, SynthesisConfig
from scipy.signal import resample_poly
from silero_vad import load_silero_vad

torch.set_num_threads(1)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
ML_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ML_DIR / "data" / "my_voice"
VOICE_PATH = ML_DIR / "models" / "tts" / "en_US-lessac-medium.onnx"
ECHO_CACHE = ML_DIR / "data" / "bargein_echo_cache"      # generated audio, never commit
OUT_DIR = ML_DIR / "results" / "week4_bargein"

# ---------------------------------------------------------
# VAD SETTINGS (copied from Backend/app/audio/vad/detector.py)
# ---------------------------------------------------------
SR = 16000
FRAME = 320                  # 20 ms WebRTC frame at 16 kHz
FRAME_MS = 20
CHUNK = 512                  # 32 ms Silero chunk
SPEECH_THRESHOLD = 0.5
START_CHUNKS = 3             # consecutive speech chunks to open a segment
END_CHUNKS = 22              # consecutive silence chunks to close it

# ---------------------------------------------------------
# EXPERIMENT SETTINGS
# ---------------------------------------------------------
THRESHOLDS = [10, 15, 20, 25, 30, 35, 40]      # frames (200-800 ms nominal)
GATES = [None, -50, -48, -46, -44, -42, -40]   # loudness gate in dBFS (None = no gate)
# -15/-20/-25 dB are the task levels; -10 dB is an extra stress case.
ECHO_LEVELS_DB = [-10, -15, -20, -25]
N_VARIANTS = 3                                  # random echo variants per reply per level
REFLECT_DELAY_MS = (20.0, 40.0)
REFLECT_GAIN = (0.25, 0.50)
NOISE_DBFS = -55.0                              # light mic noise, same for all levels
LEAD_S = 0.3                                    # silence before the echo starts
GAP_S = 0.15                                    # pause between sentences of one reply
MAX_FALSE_TRIGGER_PCT = 5.0
MIN_DETECTION_PCT = 95.0
RECOMMEND_AT_DB = -20

# Voice styles copied from Backend/app/tts/engine.py
STYLES = {
    "okay": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w_scale": 0.8,
             "volume": 1.0, "pitch_steps": 0.0},
    "upset": {"length_scale": 1.3, "noise_scale": 0.4, "noise_w_scale": 0.5,
              "volume": 0.85, "pitch_steps": -1.0},
    "low": {"length_scale": 1.4, "noise_scale": 0.4, "noise_w_scale": 0.5,
            "volume": 0.7, "pitch_steps": -2.0},
}
STYLE_ORDER = ["okay", "upset", "low"]          # replies cycle through the 3 styles

REPLIES = [
    "I hear you. That sounds really hard. Take your time, I am here with you.",
    "Thank you for telling me that. It takes courage to say it out loud. What happened next?",
    "I am listening. Tell me more about what is on your mind right now.",
    "That makes sense. Anyone in your position would feel upset. Let us take this one step at a time.",
    "I understand. You do not have to go through this alone. I am right here.",
    "It sounds like a lot is going on. Would you like to tell me what felt the heaviest today?",
    "I am sorry you are dealing with this. Your feelings matter, and I want to understand them.",
    "Let us slow down for a moment. Take a deep breath with me. Then we can keep talking.",
    "That sounds frustrating. I can hear how much this means to you. What would help most right now?",
    "You are not in trouble with me. I only want to help. Can you tell me what you need?",
    "I appreciate you being honest with me. That cannot have been easy. How are you feeling now?",
    "Thank you for staying with me. We can go as slowly as you like.",
    "I want to make sure I understand. You are saying that nobody has been listening. Is that right?",
    "That is a lot to carry. You have been strong for a long time. It is okay to feel tired.",
    "I am glad you reached out. Let us work through this together. What feels most urgent?",
    "I can tell this is important to you. I am not going anywhere. Please keep talking.",
    "It is okay to be angry. Your anger tells me something matters to you. Can you tell me what it is?",
    "Let us pause for a second. There is no rush at all. I will wait for you.",
    "I hear that you feel alone right now. I am here, and I am listening. What would you like me to know?",
    "That sounds really overwhelming. Let us break it into smaller pieces. What is the first thing on your mind?",
    "You are doing the right thing by talking about it. Many people feel this way. You are not alone.",
    "Thank you for explaining. I can see why you feel that way. What would make tonight a little easier?",
    "I am here and I am not judging you. Say whatever you need to say.",
    "It sounds like you have been hurting for a while. I am sorry. Can you tell me more about how it started?",
    "Let us take a slow breath together. In through the nose, and out through the mouth. I will stay right here.",
    "I understand you are scared. That is a normal reaction. We can face this step by step.",
    "That is a really important point. I want to give it the attention it deserves. Please go on.",
    "I am glad you are still talking with me. How is your body feeling right now? Are you able to sit down?",
    "I can hear the pain in your words. You deserve support. Is there someone you trust that we can call?",
    "You have done nothing wrong by asking for help. I am proud of you for being here. What shall we talk about first?",
]


def col(level_db):
    return f"false_trigger_pct_m{abs(level_db)}db"


def gate_label(g):
    return "none" if g is None else f"{g} dBFS"


# ---------------------------------------------------------
# LEVELS
# ---------------------------------------------------------
def frame_db(audio):
    """RMS level (dBFS) of every 20 ms frame."""
    n = len(audio) // FRAME
    fr = audio[: n * FRAME].reshape(n, FRAME)
    return 20.0 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)


def active_rms_db(x):
    """Median RMS (dBFS) of 20 ms frames within 15 dB of the loudest frame."""
    db = frame_db(x)
    act = db[db >= db.max() - 15.0]
    return float(np.median(act))


def scale_to_db(x, target_db):
    cur = active_rms_db(x)
    return (x * 10.0 ** ((target_db - cur) / 20.0)).astype(np.float32)


# ---------------------------------------------------------
# REAL USER SPEECH
# ---------------------------------------------------------
def load_real_clips():
    clips = []
    for f in sorted(DATA_DIR.glob("*/*/*.wav")):
        audio, sr = sf.read(str(f), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = resample_poly(audio, SR, sr).astype(np.float32)
        name = f"{f.parent.parent.name}/{f.parent.name}/{f.name}"
        clips.append((name, audio.astype(np.float32)))
    return clips


# ---------------------------------------------------------
# ECHO (Auralis voice)
# ---------------------------------------------------------
def clean_text(text):
    text = re.sub(r"[*_#`~>\[\]{}|]", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def synth_sentence(voice, text, style):
    cfg = SynthesisConfig(
        length_scale=style["length_scale"],
        noise_scale=style["noise_scale"],
        noise_w_scale=style["noise_w_scale"],
        volume=style["volume"],
    )
    chunks = [c.audio_float_array for c in voice.synthesize(clean_text(text), syn_config=cfg)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32)
    steps = style["pitch_steps"]
    if steps and audio.size:
        import librosa

        audio = librosa.effects.pitch_shift(audio, sr=int(voice.config.sample_rate), n_steps=steps)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def build_clean_echoes():
    """30 Auralis replies at 16 kHz, exactly as the speaker would play them."""
    ECHO_CACHE.mkdir(parents=True, exist_ok=True)
    voice = None
    out = []
    for i, reply in enumerate(REPLIES):
        style_key = STYLE_ORDER[i % len(STYLE_ORDER)]
        cache = ECHO_CACHE / f"reply_{i:02d}_{style_key}.npy"
        if cache.exists():
            out.append(np.load(cache))
            continue
        if voice is None:
            print("Loading Piper voice:", VOICE_PATH)
            voice = PiperVoice.load(str(VOICE_PATH))
        tts_sr = int(voice.config.sample_rate)
        style = STYLES[style_key]
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", reply.strip()) if s]
        pieces = []
        for s in sentences:
            pieces.append(synth_sentence(voice, s, style))
            pieces.append(np.zeros(int(GAP_S * tts_sr), dtype=np.float32))
        audio = np.concatenate(pieces)
        audio16 = resample_poly(audio, SR, tts_sr).astype(np.float32)
        np.save(cache, audio16)
        print(f"  synthesised reply {i + 1}/{len(REPLIES)} ({style_key}, {len(audio16) / SR:.1f} s)")
        out.append(audio16)
    return out


def make_echo(clean, level_db, seed):
    """Leftover echo after echo cancellation: room reflection, lower level, light noise.
    `clean` must already be scaled to the user's speech loudness (see main)."""
    rng = np.random.default_rng(seed)
    delay = int(rng.uniform(*REFLECT_DELAY_MS) / 1000.0 * SR)
    gain = rng.uniform(*REFLECT_GAIN)
    y = np.zeros(len(clean) + delay, dtype=np.float32)
    y[: len(clean)] += clean
    y[delay: delay + len(clean)] += gain * clean
    y *= 10.0 ** (level_db / 20.0)
    y = np.concatenate((np.zeros(int(LEAD_S * SR), dtype=np.float32), y))
    noise = rng.standard_normal(len(y)).astype(np.float32) * (10.0 ** (NOISE_DBFS / 20.0))
    return np.clip(y + noise, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------
# VAD SIMULATION (same logic as detector.py + processor.py)
# ---------------------------------------------------------
def simulate(model, audio):
    """
    Returns (voiced, in_seg), one value per 20 ms frame:
      voiced[i] : frame i is voiced (latest VAD probability >= threshold)
      in_seg[i] : the VAD is "in speech" at frame i (a segment is open)
    """
    model.reset_states()
    n_frames = len(audio) // FRAME
    voiced = np.zeros(n_frames, dtype=bool)
    in_seg = np.zeros(n_frames, dtype=bool)

    buf = np.empty(0, dtype=np.float32)
    in_speech = False
    cons_speech = 0
    cons_silence = 0
    last_prob = 0.0

    for fi in range(n_frames):
        buf = np.concatenate((buf, audio[fi * FRAME: (fi + 1) * FRAME]))

        while len(buf) >= CHUNK:
            chunk = buf[:CHUNK]
            buf = buf[CHUNK:]
            with torch.no_grad():
                prob = float(model(torch.from_numpy(chunk), SR).item())
            last_prob = prob

            if prob >= SPEECH_THRESHOLD:
                cons_speech += 1
                cons_silence = 0
                if not in_speech and cons_speech >= START_CHUNKS:
                    in_speech = True
            else:
                cons_speech = 0
                if in_speech:
                    cons_silence += 1
                    if cons_silence >= END_CHUNKS:
                        in_speech = False
                        cons_silence = 0

        voiced[fi] = last_prob >= SPEECH_THRESHOLD
        in_seg[fi] = in_speech

    return voiced, in_seg


def make_record(model, name, audio):
    voiced, in_seg = simulate(model, audio)
    db = frame_db(audio)[: len(voiced)]
    return {"name": name, "voiced": voiced, "in_seg": in_seg, "db": db,
            "level_db": active_rms_db(audio)}


def gated_counts(c, gate):
    """Voiced frames counted from the start of the current segment.
    With a gate, a voiced frame only counts when its level >= gate (dBFS)."""
    ok = c["voiced"] if gate is None else (c["voiced"] & (c["db"] >= gate))
    in_seg = c["in_seg"]
    counts = np.zeros(len(ok), dtype=np.int32)
    run = 0
    prev = False
    for i in range(len(ok)):
        if in_seg[i]:
            if not prev:
                run = 0
            run += int(ok[i])
            counts[i] = run
        prev = bool(in_seg[i])
    return counts


def detect_ms(counts, voiced, n):
    """Time from the first voiced frame until the segment count reaches n (None = never)."""
    if not voiced.any():
        return None
    hit = np.flatnonzero(counts >= n)
    if hit.size == 0:
        return None
    onset = int(np.argmax(voiced))
    return (int(hit[0]) - onset + 1) * FRAME_MS


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    real_clips = load_real_clips()
    if not real_clips:
        raise SystemExit(f"No wav clips found under {DATA_DIR}")
    print(f"Real speech clips: {len(real_clips)}")

    # Echo reference level = loudness of the user's own speech (AEC off)
    ref_db = float(np.median([active_rms_db(a) for _, a in real_clips]))
    print(f"Reference level (user speech, median active RMS): {ref_db:.1f} dBFS")

    print("Preparing echo (Auralis voice) ...")
    clean_echoes = [scale_to_db(c, ref_db) for c in build_clean_echoes()]

    print("Loading Silero VAD ...")
    model = load_silero_vad(onnx=True)

    # ---- real speech ----
    real = [make_record(model, n, a) for n, a in real_clips]
    print(f"Real speech simulated ({time.time() - t0:.0f} s)")

    # ---- noise-only baseline (microphone noise, nobody speaking) ----
    noise_max = []
    for k in range(30):
        rng = np.random.default_rng(5000 + k)
        noise = rng.standard_normal(6 * SR).astype(np.float32) * (10.0 ** (NOISE_DBFS / 20.0))
        rec = make_record(model, f"noise{k}", noise)
        noise_max.append(int(gated_counts(rec, None).max()))
    print(f"Noise-only baseline ({NOISE_DBFS:.0f} dBFS): max voiced count in any clip = {max(noise_max)}")

    # ---- echo ----
    echo = {lvl: [] for lvl in ECHO_LEVELS_DB}
    for lvl in ECHO_LEVELS_DB:
        for r, clean in enumerate(clean_echoes):
            for k in range(N_VARIANTS):
                audio = make_echo(clean, lvl, seed=1000 * r + k)
                echo[lvl].append(make_record(model, f"reply{r:02d}_v{k}", audio))
        print(f"Echo {lvl} dB simulated: {len(echo[lvl])} clips ({time.time() - t0:.0f} s)")

    # ---- level / voiced-frame check ----
    print("\nLevel check (median active RMS dBFS | median voiced frames per clip):")
    print(f"  real user speech : {np.median([c['level_db'] for c in real]):6.1f} | "
          f"{np.median([int(c['voiced'].sum()) for c in real]):5.0f}")
    for lvl in ECHO_LEVELS_DB:
        print(f"  echo at {lvl:>3} dB  : {np.median([c['level_db'] for c in echo[lvl]]):6.1f} | "
              f"{np.median([int(c['voiced'].sum()) for c in echo[lvl]]):5.0f}")

    # ---- metrics: every (gate, threshold) pair ----
    rows = []
    for g in GATES:
        real_counts = [gated_counts(c, g) for c in real]
        echo_max = {lvl: np.array([int(gated_counts(c, g).max()) for c in echo[lvl]])
                    for lvl in ECHO_LEVELS_DB}
        for n in THRESHOLDS:
            times = [detect_ms(cn, c["voiced"], n) for cn, c in zip(real_counts, real)]
            ok = [t for t in times if t is not None]
            row = {
                "gate_label": gate_label(g),
                "gate_dbfs": np.nan if g is None else g,
                "gate_vs_user_db": np.nan if g is None else round(g - ref_db, 1),
                "threshold_frames": n,
                "threshold_ms": n * FRAME_MS,
                "detection_rate_pct": 100.0 * len(ok) / len(real),
                "mean_detect_ms": float(np.mean(ok)) if ok else float("nan"),
                "median_detect_ms": float(np.median(ok)) if ok else float("nan"),
            }
            for lvl in ECHO_LEVELS_DB:
                row[col(lvl)] = 100.0 * float(np.mean(echo_max[lvl] >= n))
            rows.append(row)

    df = pd.DataFrame(rows)
    ft_col = col(RECOMMEND_AT_DB)
    df["meets_criteria"] = (df[ft_col] <= MAX_FALSE_TRIGGER_PCT) & \
                           (df["detection_rate_pct"] >= MIN_DETECTION_PCT)
    df["ft_sum_task_levels"] = df[col(-15)] + df[col(-20)] + df[col(-25)]

    # ---- recommendation ----
    cand = df[df["meets_criteria"]].sort_values(
        ["threshold_frames", "ft_sum_task_levels", "detection_rate_pct"],
        ascending=[True, True, False])
    rec_row = cand.iloc[0] if not cand.empty else None
    df["recommended"] = False
    if rec_row is not None:
        df.loc[(df["gate_label"] == rec_row["gate_label"]) &
               (df["threshold_frames"] == rec_row["threshold_frames"]), "recommended"] = True

    # ---- save CSV ----
    csv_path = OUT_DIR / "barge_in_thresholds.csv"
    out = df.drop(columns=["ft_sum_task_levels"]).copy()
    out["n_real_clips"] = len(real)
    out["n_echo_clips_per_level"] = len(echo[ECHO_LEVELS_DB[0]])
    out["user_speech_dbfs"] = round(ref_db, 1)
    out.round(2).to_csv(csv_path, index=False)

    # ---- plot: detection time vs false trigger rate ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, lvl in zip(axes, (-20, -15)):
        c_ = col(lvl)
        for g in GATES:
            d = df[df["gate_label"] == gate_label(g)]
            ax.plot(d[c_], d["mean_detect_ms"], marker="o", markersize=4, linewidth=1.2,
                    label=gate_label(g))
        if rec_row is not None:
            d = df[df["gate_label"] == rec_row["gate_label"]]
            for _, r in d.iterrows():
                ax.annotate(str(int(r["threshold_frames"])), (r[c_], r["mean_detect_ms"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=8)
            ax.scatter([rec_row[c_]], [rec_row["mean_detect_ms"]], marker="*", s=320,
                       color="gold", edgecolor="black", zorder=6,
                       label=f"recommended: N={int(rec_row['threshold_frames'])}, "
                             f"gate {rec_row['gate_label']}")
        ax.axvline(MAX_FALSE_TRIGGER_PCT, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel(f"False trigger rate on echo at {lvl} dB (%)")
        ax.set_xlim(-3, 103)
        ax.grid(alpha=0.3)
        ax.set_title(f"Echo at {lvl} dB" + ("  (criteria level)" if lvl == RECOMMEND_AT_DB else ""))
    axes[0].set_ylabel("Mean time to detect real user speech (ms)")
    axes[1].legend(title="loudness gate (numbers on line = N frames)", fontsize=8, loc="lower left")
    fig.suptitle("Barge-in trade-off: detection time vs false triggers "
                 "(red line = 5% false-trigger limit)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "barge_in_tradeoff.png", dpi=150)
    plt.close(fig)

    # ---- print ----
    print("\n=== 1) FRAME COUNT ONLY (no loudness gate) ===")
    print(f"real clips: {len(real)} | echo clips per level: {len(echo[ECHO_LEVELS_DB[0]])} | "
          f"echo reference = user speech level ({ref_db:.1f} dBFS)")
    print("(mean/median_ms = time from first voiced frame; FT = false trigger rate on echo;")
    print(" -10 dB is an extra stress case beyond the -15/-20/-25 dB task levels)\n")
    ng = df[df["gate_label"] == "none"][["threshold_frames", "threshold_ms", "detection_rate_pct",
                                         "mean_detect_ms", "median_detect_ms"] +
                                        [col(l) for l in ECHO_LEVELS_DB]].round(1)
    ng.columns = ["frames", "ms", "detect%", "mean_ms", "median_ms"] + \
                 [f"FT%@{l}dB" for l in ECHO_LEVELS_DB]
    print(ng.to_string(index=False))

    def matrix(value):
        m = df.pivot(index="gate_label", columns="threshold_frames", values=value)
        m = m.loc[[gate_label(g) for g in GATES]].round(1)
        m.index.name = "gate \\ N frames"
        m.columns.name = None
        return m

    print("\n=== 2) WITH LOUDNESS GATE (rows = gate, columns = N frames) ===")
    print(f"(user speech level = {ref_db:.1f} dBFS; gate -42 dBFS = {-42 - ref_db:.1f} dB vs user)")
    print("\nDetection of real user speech (%):")
    print(matrix("detection_rate_pct").to_string())
    print("\nFalse triggers at -20 dB (%):")
    print(matrix(col(-20)).to_string())
    print("\nFalse triggers at -15 dB (%):")
    print(matrix(col(-15)).to_string())
    print("\nFalse triggers at -25 dB (%):")
    print(matrix(col(-25)).to_string())
    print("\nMean time to detect real speech (ms):")
    print(matrix("mean_detect_ms").to_string())

    print("\n=== RECOMMENDATION ===")
    best_ng = df[df["gate_label"] == "none"].sort_values(ft_col).iloc[0]
    if not df[(df["gate_label"] == "none") & df["meets_criteria"]].empty:
        print("Frame count alone is enough (see table 1).")
    else:
        print(f"Frame count alone: NO threshold meets the criteria "
              f"(lowest false-trigger rate at {RECOMMEND_AT_DB} dB is "
              f"{best_ng[ft_col]:.1f}% at N={int(best_ng['threshold_frames'])}).")
    if rec_row is None:
        print("With a loudness gate: NO combination meets the criteria either.")
        print("BARGE_IN_MIN_VOICED_FRAMES = no safe value found")
    else:
        n_rec = int(rec_row["threshold_frames"])
        print(f"With a loudness gate: BARGE_IN_MIN_VOICED_FRAMES = {n_rec}, "
              f"counting only voiced frames louder than {rec_row['gate_label']}")
        print(f"  (gate = {rec_row['gate_vs_user_db']:+.1f} dB relative to the user's typical "
              f"speech level of {ref_db:.1f} dBFS)")
        print(f"  detection {rec_row['detection_rate_pct']:.1f}% | "
              f"mean {rec_row['mean_detect_ms']:.0f} ms | median {rec_row['median_detect_ms']:.0f} ms")
        print("  false triggers: " + ", ".join(
            f"{rec_row[col(l)]:.1f}% @{l} dB" for l in (-15, -20, -25)) +
            f" (stress case {rec_row[col(-10)]:.1f}% @-10 dB)")
    cur = df[(df["gate_label"] == "none") & (df["threshold_frames"] == 20)].iloc[0]
    print(f"  current backend default (N=20, no gate): detection {cur['detection_rate_pct']:.1f}% | "
          f"mean {cur['mean_detect_ms']:.0f} ms | FT " +
          ", ".join(f"{cur[col(l)]:.1f}% @{l} dB" for l in (-15, -20, -25)))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {OUT_DIR / 'barge_in_tradeoff.png'}")
    print(f"Done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
