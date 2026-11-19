"""
ML Week 3 Day 5: text emotion model + voice/text fusion.

Run from the ml folder:
    python src\\models\\text_emotion.py
"""

import csv
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\Auralis-ML-Work\hf-cache")

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
MODEL_TO_OURS = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    # "surprise" is dropped on purpose
}
# CHECK against emotion_engine.py in the backend; adjust if the real groups differ
GROUPS = {
    "angry": "upset", "disgust": "upset", "fear": "upset",
    "sad": "low",
    "neutral": "okay", "happy": "okay",
}

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "week3_fusion"

# ---------------------------------------------------------------- Task 1
_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline(
            "text-classification", model=MODEL_NAME, top_k=None, device=-1
        )
    return _pipe


def predict_text(text):
    """Return {label: probability} over our 6 labels (sums to 1)."""
    out = _get_pipe()(text)
    if out and isinstance(out[0], list):
        out = out[0]
    scores = {
        MODEL_TO_OURS[d["label"]]: float(d["score"])
        for d in out
        if d["label"] in MODEL_TO_OURS
    }
    total = sum(scores.values()) or 1.0
    return {k: scores.get(k, 0.0) / total for k in LABELS}


# ---------------------------------------------------------------- Task 3
def fuse(voice_probs, text_probs):
    """Return (final_label, confidence, source) where source is 'voice' or 'words'."""
    t_label = max(text_probs, key=text_probs.get)
    t_conf = text_probs[t_label]

    if t_label != "neutral" and t_conf >= 0.6:
        w_text, w_voice, source = 0.7, 0.3, "words"
    else:
        w_text, w_voice, source = 0.2, 0.8, "voice"

    final = {k: w_text * text_probs[k] + w_voice * voice_probs[k] for k in LABELS}
    label = max(final, key=final.get)
    return label, final[label], source


def make_voice_probs(label, conf):
    rest = (1.0 - conf) / (len(LABELS) - 1)
    return {k: (conf if k == label else rest) for k in LABELS}


# ---------------------------------------------------------------- Task 2 data
TEST_SET = {
    "angry": [
        "I can't believe they cancelled my exam again.",
        "I told them three times and nobody fixed it, I'm so furious.",
        "Stop interrupting me, I'm sick of this.",
        "My roommate ate my food again and I'm really angry.",
        "This is completely unfair and I'm fed up.",
        "I'm so mad at my manager right now.",
        "They lied to me and I'm not letting it go.",
        "Why does this app keep crashing, it's driving me crazy.",
        "I'm really annoyed that nobody told me about the change.",
        "I am so angry that he broke his promise again.",
    ],
    "disgust": [
        "That smell from the kitchen is absolutely revolting.",
        "I found mold in my lunchbox and it was disgusting.",
        "The way he talks about people makes me feel sick.",
        "Ugh, the bathroom at that place was filthy.",
        "I can't stand watching people chew with their mouth open.",
        "The milk had gone bad and the whole fridge stinks, gross.",
        "I'm disgusted by how they treated the staff.",
        "That video of the dirty kitchen made my stomach turn.",
        "It's repulsive how some people litter everywhere.",
        "I saw a cockroach in the canteen, that was nasty.",
    ],
    "fear": [
        "I think someone is following me home.",
        "I'm really scared about my results tomorrow.",
        "I heard a strange noise downstairs and I'm terrified.",
        "I'm so nervous about the interview, my hands are shaking.",
        "What if I fail and everyone finds out? I'm really afraid.",
        "I'm worried something bad is going to happen.",
        "The storm is getting louder and I'm frightened.",
        "I'm anxious about going on stage in front of everyone.",
        "I'm scared to tell my parents about my marks.",
        "I get really panicky whenever I have to drive on the highway.",
    ],
    "happy": [
        "I just got the internship and I'm so excited.",
        "Today was honestly the best day of my life.",
        "I'm really happy, my friends organised a party for me.",
        "I passed my exam and I can't stop smiling.",
        "Everything is going great and I feel wonderful.",
        "I'm so glad you called, it made my whole day.",
        "We won the match and the whole team is celebrating.",
        "I got the job offer and I'm thrilled.",
        "I love this song, it makes me so happy.",
        "I feel so grateful and cheerful today.",
    ],
    "neutral": [
        "I need to buy some groceries after class.",
        "The bus usually arrives around nine.",
        "Can you tell me what the weather is like tomorrow?",
        "I have a lecture on networks this afternoon.",
        "I'm going to the library to return a book.",
        "The report is due on Friday.",
        "I usually have tea in the morning.",
        "My laptop has sixteen gigabytes of memory.",
        "We are meeting in the cafeteria at noon.",
        "I'll check the schedule and get back to you.",
    ],
    "sad": [
        "I feel so alone these days.",
        "I miss my grandmother so much.",
        "Nothing seems to cheer me up lately.",
        "I failed the test and I feel really down.",
        "My best friend moved away and the house feels empty.",
        "I've been crying all evening and I feel terrible.",
        "I feel like I let everyone down.",
        "It's been a really hard week and I'm exhausted and low.",
        "I'm heartbroken that it ended this way.",
        "I just feel empty and sad today.",
    ],
}

# Paste the 10 neutral sentences from src\data\record_my_voice.py here
NEUTRAL_RECORDING = [
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

# (transcript, voice label, voice confidence, expected final)
FUSION_CASES = [
    ("I'm very sad today.", "neutral", 0.93, "sad"),
    ("Today my girlfriend cheated on me, so I'm very sad today.", "disgust", 0.59, "sad"),
    ("I am very angry today.", "angry", 0.55, "angry"),
    ("Hi, my name is Ankit and today is my best day. I'm very happy.", "happy", 0.83, "happy"),
    ("I think someone is knocking my door.", "disgust", 0.66, "fear"),
    ("The meeting is at five o'clock.", "angry", 0.80, "angry"),
]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"saved {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading text model (first run downloads it)...")
    for _ in range(3):
        predict_text("warm up")

    # ---- Task 2: accuracy, confusion matrix, latency
    sentences = [(s, emo) for emo, items in TEST_SET.items() for s in items]
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = [[0] * len(LABELS) for _ in LABELS]
    latencies = []
    correct = 0
    group_correct = 0

    for text, true in sentences:
        t0 = time.perf_counter()
        probs = predict_text(text)
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = max(probs, key=probs.get)
        cm[idx[true]][idx[pred]] += 1
        correct += pred == true
        group_correct += GROUPS[pred] == GROUPS[true]

    n = len(sentences)
    acc = correct / n
    group_acc = group_correct / n
    median_ms = statistics.median(latencies)

    summary = [
        ("sentences", n),
        ("text_only_accuracy", round(acc, 4)),
        ("three_group_accuracy", round(group_acc, 4)),
        ("median_latency_ms_cpu", round(median_ms, 1)),
        ("latency_under_50ms", median_ms < 50),
    ]
    for i, lab in enumerate(LABELS):
        row_total = sum(cm[i])
        summary.append((f"accuracy_{lab}", round(cm[i][i] / row_total, 4) if row_total else 0))

    if NEUTRAL_RECORDING:
        n_neutral = 0
        for s in NEUTRAL_RECORDING:
            p = predict_text(s)
            n_neutral += max(p, key=p.get) == "neutral"
        summary.append(("recording_neutral_sentences", len(NEUTRAL_RECORDING)))
        summary.append(("recording_predicted_neutral", n_neutral))
        print(f"Recording sentences predicted neutral: {n_neutral}/{len(NEUTRAL_RECORDING)}")
    else:
        print("NEUTRAL_RECORDING is empty - paste the 10 sentences and run again")

    write_csv(OUT_DIR / "text_accuracy_summary.csv", ["metric", "value"], summary)
    write_csv(
        OUT_DIR / "text_confusion_matrix.csv",
        ["true\\pred"] + LABELS,
        [[LABELS[i]] + cm[i] for i in range(len(LABELS))],
    )
    print(f"text accuracy {acc:.1%}, 3-group {group_acc:.1%}, median latency {median_ms:.1f} ms")

    # ---- Task 3: fusion cases
    rows = []
    n_ok = 0
    for transcript, v_label, v_conf, expected in FUSION_CASES:
        t_probs = predict_text(transcript)
        v_probs = make_voice_probs(v_label, v_conf)
        t_label = max(t_probs, key=t_probs.get)
        final, conf, source = fuse(v_probs, t_probs)
        ok = final == expected
        n_ok += ok
        rows.append([transcript, f"{v_label} {v_conf:.0%}", f"{t_label} {t_probs[t_label]:.0%}",
                     final, f"{conf:.2f}", source, expected, ok])
        print(f"{'OK ' if ok else 'BAD'} | {transcript[:45]:45s} | text={t_label} | final={final} ({source})")

    write_csv(
        OUT_DIR / "fusion_test_cases.csv",
        ["transcript", "voice_label", "text_label", "final_label", "final_conf",
         "source", "expected", "correct"],
        rows,
    )
    print(f"fusion cases correct: {n_ok}/{len(FUSION_CASES)}")


if __name__ == "__main__":
    main()