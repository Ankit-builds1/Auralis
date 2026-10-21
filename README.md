# Auralis — ML

Real-time Voice-to-Voice Emotion Engine: **machine learning module**.

Speech emotion recognition from raw audio, emotion from the words, the fusion of both, the emotion-aware Llama prompt, and the text-to-speech research behind Auralis's voice.

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2).

| Role | Owner | Branch |
|---|---|---|
| **ML: emotion recognition, fusion, TTS research (this module)** | Siddhant (Weeks 1–2), then Ankit Dash + Priya Nirmal (Week 3) | `ML` |
| Backend: WebRTC, VAD, STT, LLM + TTS streaming | Priya Nirmal | `Backend` |
| Frontend: WebRTC client + live dashboard | Ankit Dash | `frontend` |

---

## What this module does (deployed, Week 3)

```
             speech segment (16 kHz audio)                 transcript (Whisper)
                          │                                        │
                          ▼                                        ▼
        WavLM-Large (frozen, layer 11, mean-pooled)     DistilRoBERTa emotion model
                          │                             (j-hartmann, 6 of 7 labels)
                          ▼                                        │
     StandardScaler → Logistic Regression                          │
     (CREMA-D + our own voices, "adapted")                         │
                          │                                        │
                          └──────────►   FUSION   ◄────────────────┘
                                 words decide if the text model is
                                 ≥ 60 % sure of a non-neutral emotion,
                                 otherwise the voice decides
                                            │
                                            ▼
                         emotion + group (upset / low / okay)
                                            │
                   ┌────────────────────────┴───────────────────────┐
                   ▼                                                ▼
     emotion-aware Llama 3.2 prompt                    Piper voice style
     (emotion = possibly-imperfect context)            calm / soft / normal
```

Labels: `angry, disgust, fear, happy, neutral, sad`. Groups used by the backend: **upset** = angry/disgust/fear, **low** = sad, **okay** = happy/neutral.

---

## Week 3 results at a glance

| Day | Question | Answer |
|---|---|---|
| 1 | Which TTS is fast enough on CPU? | **Piper**: 31–175 ms per sentence (RTF ≈ 0.04) vs Kokoro 422–1887 ms (RTF ≈ 0.4) |
| 2 | Can a pretrained speech model beat hand-made features? | **WavLM-Large: 74.6 % test** (was 49.0 % with the MFCC SVM), 3-group 80.7 % |
| 3 | Why is it wrong on our voices, and can we fix it? | Domain shift: **19.4 % → 52.8 %** on our held-out clips after adding 120 of our own recordings |
| 4 | Do more datasets help new speakers? | Yes for new studio voices (RAVDESS 59 % → 78 %), **no for our voices** (56 % → 42 %), so the adapted model stays deployed |
| 5 | Can the words fix what the voice misses? | Text model 90 % on 60 sentences, 27 ms; live test **voice-only 29 % → fused 71 %** |
| 6 | Can a zero-shot TTS carry emotion? | XTTS-v2 on GPU: first audio **512 ms**, emotion heard **33 %** vs Piper **0 %** (chance 17 %) |

---

## Datasets

| Dataset | Clips used | Speakers | Used for | Licence |
|---|---|---|---|---|
| **CREMA-D** | 7,442 | 91 actors | main training / test (actor-independent splits) | ODbL |
| **RAVDESS** (speech) | 1,248 (surprised dropped, calm → neutral) | 24 actors | Week 3 Day 4 multi-corpus test | CC BY-NC-SA 4.0 |
| **TESS** | 2,400 (pleasant surprise dropped) | 2 actresses | Week 3 Day 4 multi-corpus test | CC BY-NC-ND 4.0 |
| **Our voices** | 120 (2 speakers × 6 emotions × 10 sentences, 4 s each) | Ankit, Priya | Week 3 Day 3 speaker adaptation | private, **never pushed to git** |

CREMA-D splits are **actor-independent** (`GroupShuffleSplit` by `actor_id`, `random_state=42`): train 5,152 / validation 1,142 / test 1,148 clips (63 / 14 / 14 actors), so every test score is on voices the model never heard.

Our recordings use the **same 10 neutral sentences in all 6 emotions** (CREMA-D style), so the model has to learn the tone, not the words.

---

## Week 3 in detail

### Day 1 — TTS benchmark (CPU)

`src/tts/benchmark_tts.py` → `results/week3_tts/tts_benchmark.csv`

| Engine | Load | Synth time (3 / 10 / 22 words) | Real-time factor |
|---|---|---|---|
| **Piper** `en_US-lessac-medium` | 3.2 s | 31 / 93 / 175 ms | **0.03–0.04** |
| Kokoro 82M (int8 ONNX) | 1.2 s | 422 / 849 / 1887 ms | 0.34–0.46 |

**Decision:** Piper, about 10× faster, so the first sentence can be spoken almost immediately. The LLM was switched to `llama3.2:3b` for the same reason.

### Day 2 — WavLM-Large emotion model

`src/features/extract_ssl_embeddings.py`, `src/models/train_ssl_classifier.py`, `src/models/ssl_inference.py`

Each clip is passed through a **frozen** self-supervised speech model. Every hidden layer is averaged over time, and a logistic regression is trained on the best layer (chosen on validation only).

| Model | Best layer | Val acc | **Test acc** | Test macro-F1 | 3-group |
|---|---|---|---|---|---|
| Temporal MFCC SVM (Week 2) | – | 0.513 | 0.490 | 0.482 | – |
| WavLM-Base-Plus | 5 | 0.750 | 0.693 | 0.694 | 0.779 |
| **WavLM-Large** | **11** | **0.772** | **0.746** | **0.747** | **0.808** |

Live inference: about 0.4 s per segment on CPU, running in parallel with Whisper. `src/tts/emotion_style.py` maps the emotion group to a Piper voice style (upset → calm: slower, −1 semitone; low → soft: slower, −2 semitones, quieter; okay → normal).

### Day 3 — Speaker adaptation (our own voices)

`src/data/record_my_voice.py` → `src/features/extract_my_voice.py` → `src/models/train_speaker_adapted.py`

The live test showed the CREMA-D model was **confidently wrong** on our voices (Indian English, laptop mic, natural speech). We recorded 120 clips and retrained the classifier on CREMA-D + our clips (our clips weighted 30×). A fair test: 30 % of our clips (36) were held out, stratified by speaker × emotion.

| Test set | Clips | Before | **After** | 3-group before → after |
|---|---|---|---|---|
| **Our voices (held out)** | 36 | 19.4 % | **52.8 %** | 38.9 % → 61.1 % |
| Ankit | 18 | 22.2 % | 55.6 % | 27.8 % → 55.6 % |
| Priya | 18 | 16.7 % | 50.0 % | 50.0 % → 66.7 % |
| CREMA-D test | 1,148 | 74.6 % | 74.0 % | 80.7 % → 80.3 % |

Adding a second speaker (Priya) also raised Ankit's own score (44 % → 56 %): the model learns the accent and mic, not just one person.
**Deployed** as `ssl_wavlm-large-adapted_emotion.joblib` (`EMOTION_MODEL=wavlm-large-adapted`, live confidence gate 0.6).

### Day 4 — Multi-corpus training + leave-one-corpus-out

`src/features/extract_corpus.py`, `src/models/train_multicorpus.py` → `results/week3_multi/`

**Speaker-held-out test** (old = CREMA-D + ours, new = CREMA-D + RAVDESS + TESS + ours):

| Test set | Clips | Old | New |
|---|---|---|---|
| CREMA-D test actors | 1,148 | 73.5 % | 72.1 % |
| RAVDESS actors 21–24 (new people) | 208 | 58.7 % | **78.4 %** |
| TESS 20 % (same 2 voices, optimistic) | 480 | 56.9 % | 99.6 % |
| **Our voices (held out)** | 36 | **55.6 %** | 41.7 % |

**Leave-one-corpus-out** (a whole dataset never seen in training, the closest thing to "a new user"):

| Unseen corpus | Trained on | Accuracy | 3-group |
|---|---|---|---|
| RAVDESS | CREMA-D + TESS + ours | 64.4 % | 75.8 % |
| TESS | CREMA-D + RAVDESS + ours | 65.9 % | 66.5 % (was 57.7 % without RAVDESS) |
| CREMA-D | RAVDESS + TESS + ours | 38.1 % | 49.8 % |

**Decision:** more corpora help new **studio** speakers, but acted North-American speech pulls the model away from our real voices (raising our weight to 60× did not fix it). The **adapted model stays deployed**; the multi-corpus model is a documented result.

### Day 5 — Text emotion + voice/text fusion

`src/models/text_emotion.py` → `results/week3_fusion/`

Model: `j-hartmann/emotion-english-distilroberta-base` (7 labels; surprise dropped, rest renormalised).

| Metric | Result |
|---|---|
| Accuracy on 60 conversational sentences | **90.0 %** (3-group 93.3 %) |
| Median latency (CPU) | **27.4 ms** |
| Our 10 neutral recording sentences predicted neutral | 9 / 10 (it doesn't override the voice when words carry no emotion) |
| Fusion test cases correct | 5 / 6 |

**Fusion rule:** if the text model's top label is not neutral and ≥ 0.6 → `0.7 × text + 0.3 × voice` (words decide), otherwise `0.2 × text + 0.8 × voice` (voice decides).

**Live result** (frontend Emotion Test Mode, 7 acted turns): voice-only **29 %**, words-only **71 %**, final fused **71 %** (3-group 86 %); 6 of 7 turns were decided by the words. Example: *"I'm very sad today"*, voice said neutral 93 %, words sad 99 %, final **sad**.

### Day 6 — Zero-shot emotional TTS (XTTS-v2, GPU)

`src/tts/xtts_emotion_test.py` (separate TTS venv) → `src/tts/score_tts_emotion.py` (ML venv) → `results/week3_tts/`

The project plan asks for a zero-shot TTS conditioned on an emotional baseline (calm vs panicked). XTTS-v2 was conditioned on our recorded emotion clips (neutral → calm, sad → soft, fear → panicked, angry, happy); the same 3 sentences were spoken in every style.

| Metric (RTX 2050, 4 GB) | XTTS-v2 | Piper (current) |
|---|---|---|
| First audio chunk (streaming) | **512 ms** median (max 631) | ~100–400 ms |
| Real-time factor | 0.65 (streamable) | ~0.04 |
| GPU memory | 1.9 GB | 0 (CPU) |

**Does the voice carry the emotion?** Our emotion model listened to every file:

| Engine | Clips | Intended emotion heard | 3-group | Mean p(intended) |
|---|---|---|---|---|
| **XTTS-v2** | 15 | **33 %** | 47 % | **33 %** |
| Piper styles (speed/pitch) | 4 | 0 % | 50 % | 9 % |
| Chance | – | 17 % | 33 % | 17 % |

"Panicked" (fear) transferred best (99 % and 92 %); happy and calm were weak. **Honest listening note:** the styles sound only slightly different by ear. The reference clips were short (4 s) and quietly acted, and speed was not varied. **Not deployed yet:** Piper remains the live voice; XTTS needs a GPU, shares 4 GB with Llama, and would clone a fixed "Auralis" reference voice (not the user's).

---

## Earlier work (Weeks 1–2, Siddhant)

| Experiment | Features | Model | Val acc | Test acc | Test macro-F1 |
|---|---|---|---|---|---|
| Baseline | MFCC mean (40) | Random Forest | 0.470 | 0.414 | — |
| Combined temporal + prosodic | temporal + RMS, ZCR, spectral | SVM RBF (C=10, balanced) | 0.528 | 0.503 | 0.498 |
| Temporal (deployed in Week 2) | MFCC + Δ + ΔΔ, 10 regions (2,400) | SVM RBF | 0.513 | 0.490 | 0.482 |

Robustness (combined model): clean 0.503, Gaussian noise 0.195; noise-augmented training raised noisy accuracy to 0.443 (clean 0.469). These results motivated the switch to WavLM in Week 3.

---

## Getting Started

```bash
git clone https://github.com/Ankit-builds1/Auralis.git
cd Auralis
git checkout ML
cd ml

python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install -r requirements.txt
```

Set the Hugging Face cache to a big drive first (WavLM-Large is 1.2 GB):

```powershell
$env:HF_HOME = "D:\path\to\hf-cache"
```

**Not in git** (too large or private), shared manually:

| File / folder | What |
|---|---|
| `models/ssl_wavlm-large_emotion.joblib` | CREMA-D WavLM classifier (~51 KB) |
| `models/ssl_wavlm-large-adapted_emotion.joblib` | **deployed** adapted classifier (~51 KB) |
| `data/features/ssl/*.npz` | WavLM embeddings (regenerate with the scripts) |
| `data/my_voice/` | our voice recordings (private) |
| `data/raw/ravdess/`, `data/raw/tess/` | download from Zenodo / Kaggle |
| `results/week3_tts/xtts/` | XTTS clones of our voice (private) |

### Reproduce Week 3 (from the `ml/` folder)

```powershell
# Day 2: WavLM embeddings + classifier
python -m src.features.extract_ssl_embeddings --model wavlm-large
python -m src.models.train_ssl_classifier --model wavlm-large

# Day 3: record your voice, then adapt
python -m src.data.record_my_voice --speaker <name>
python -m src.features.extract_my_voice --model wavlm-large
python -m src.models.train_speaker_adapted --model wavlm-large --weight 30

# Day 4: extra corpora (extract into data/raw/ravdess and data/raw/tess first)
python -m src.features.extract_corpus --corpus ravdess --model wavlm-large
python -m src.features.extract_corpus --corpus tess --model wavlm-large
python -m src.models.train_multicorpus --model wavlm-large --weight 60

# Day 5: text emotion + fusion
python src\models\text_emotion.py

# Day 6: XTTS (separate venv: torch 2.8.0+cu126, coqui-tts 0.28), then score in the ML venv
python -m src.tts.xtts_emotion_test --speaker <name>
python -m src.tts.score_tts_emotion
```

---

## Project Structure

```
ml/
├── data/
│   ├── raw/                      # CREMA-D (+ RAVDESS, TESS: gitignored)
│   ├── metadata/                 # labels.csv + actor-independent splits
│   ├── my_voice/                 # our recordings (gitignored)
│   └── features/                 # generated features (gitignored)
├── models/                       # .joblib classifiers (gitignored)
├── results/
│   ├── week3_ssl/                # WavLM layer search + test results
│   ├── week3_adapt/              # speaker adaptation
│   ├── week3_multi/              # multi-corpus + leave-one-corpus-out
│   ├── week3_fusion/             # text model + fusion cases
│   └── week3_tts/                # Piper/Kokoro benchmark, XTTS benchmark + emotion scores
├── src/
│   ├── data/                     # metadata, splits, voice recorder
│   ├── features/                 # MFCC/prosodic (Wk 1–2), WavLM embeddings, corpora (Wk 3)
│   ├── models/                   # SVMs (Wk 1–2), WavLM classifier, adaptation, multi-corpus, text emotion
│   ├── tts/                      # TTS benchmark, voice styles, XTTS test + scoring
│   ├── llm/                      # persona prompt + streaming Ollama client
│   └── pipeline.py               # audio + transcript → emotion → reply (Week 2)
└── requirements.txt
```

---

## Progress

**Week 1 ✅** CREMA-D exploration, metadata, actor-independent splits, MFCC baseline

**Week 2 ✅** Advanced, prosodic and temporal features; SVM tuning; robustness + noise augmentation; streaming Llama client + emotion-aware prompt

**Week 3 ✅** Piper chosen (Day 1) · WavLM-Large 74.6 % (Day 2) · speaker adaptation 19 % → 53 % on our voices (Day 3) · multi-corpus + leave-one-corpus-out (Day 4) · text emotion + fusion, live 29 % → 71 % (Day 5) · XTTS-v2 emotional TTS study (Day 6) · this report (Day 7)

**Week 4 (planned)** Interruption handling (barge-in) support, XTTS reference-voice and speed experiment, more speakers for adaptation

---

## Known Limitations

- **Voice-only emotion on real speakers is ~50–55 %** (6 classes). The words carry most of the signal in live use, so fusion is essential.
- **Small private test set:** 36 held-out clips from 2 speakers; one clip ≈ 2.8 %. Numbers are indicative.
- **Acted data:** CREMA-D, RAVDESS and TESS are acted studio speech; natural conversation is subtler.
- **Disgust is over-predicted** live; sad is often missed by the voice (the words fix most of these).
- **Text model:** 90 % on sentences with clear emotion words; neutral sentences are only 70 % correct.
- **XTTS-v2:** emotion transfer is subtle by ear; it needs a GPU, and its model weights are non-commercial (Coqui Public Model License), which is fine for this internship but not for a product.
- Dataset licences (RAVDESS, TESS) are non-commercial.
