# Auralis — ML

Real-time Voice-to-Voice Emotion Engine: **machine learning module**.

Speech emotion recognition from raw audio, plus the Llama prompt and streaming client that turn a transcript and a detected emotion into an empathetic reply.

Part of Infotact Solutions' Advanced Generative AI Engineering internship (Project 2).

| Role | Owner | Branch |
|---|---|---|
| **ML: emotion recognition + LLM prompts (this module)** | Siddhant | `ML` |
| Backend: WebRTC, VAD, STT, LLM streaming | Priya Nirmal | `Backend` |
| Frontend: WebRTC client + live dashboard | Ankit Dash | `frontend` |

---

## What this module does

```
Speech segment (audio)            Transcript (from Whisper)
        │                                  │
        ▼                                  │
 Temporal feature extraction               │
 (40 MFCC + Δ + ΔΔ, 10 time regions,       │
  mean + std → 2,400 features)             │
        │                                  │
        ▼                                  │
 StandardScaler → SVM (RBF)                │
        │                                  │
        ▼                                  ▼
 emotion label ──────────────►  Emotion-aware Llama prompt
 (angry / disgust / fear /                 │
  happy / neutral / sad)                   ▼
                                Streamed, empathetic 1–2 sentence reply
```

---

## Dataset: CREMA-D

| Item | Value |
|---|---|
| Clips | **7,442** |
| Actors | **91** |
| Emotions | angry, disgust, fear, happy, sad (1,271 each), neutral (1,087) |
| Audio | resampled to 16 kHz mono |

**Actor-independent splits** (`GroupShuffleSplit` by `actor_id`, `random_state=42`), so the model is always tested on **voices it has never heard**:

| Split | Clips | Actors |
|---|---|---|
| Train | 5,152 | 63 |
| Validation | 1,142 | 14 |
| Test | 1,148 | 14 |

---

## Experiments

| Experiment | Features | Model | Val acc | Test acc | Test macro-F1 |
|---|---|---|---|---|---|
| Baseline | MFCC mean (40) | Random Forest | 0.470 | 0.414 | — |
| Advanced features | MFCC + Δ + ΔΔ + spectral + energy (246) | Random Forest | — | — | — |
| Combined temporal + prosodic | temporal + RMS, ZCR, spectral centroid/bandwidth/rolloff/contrast | SVM RBF (C=10, balanced) | **0.528** | **0.503** | **0.498** |
| Model comparison | combined | SVM RBF / SVM linear | 0.490 / 0.468 | 0.477 / 0.476 | 0.473 / 0.474 |
| **Temporal (deployed)** | **MFCC + Δ + ΔΔ, 10 regions (2,400)** | **SVM RBF** | **0.513** | **0.490** | **0.482** |

Chance level for 6 classes is **16.7 %**.

**Tuning:** SVM `C` ∈ {1, 5, 10, 20, 50} (best ≥ 10); class weights `balanced` vs custom (no meaningful gain from custom weights).

### Per-emotion results (deployed model, test set)

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| angry | 0.54 | **0.77** | **0.63** |
| neutral | 0.57 | 0.55 | 0.56 |
| sad | 0.51 | 0.47 | 0.49 |
| happy | 0.42 | 0.50 | 0.46 |
| fear | 0.43 | 0.37 | 0.40 |
| disgust | 0.44 | 0.29 | 0.35 |

Anger is detected best; disgust and fear are most often confused with other emotions.

---

## Robustness

Tested on the combined model (clean test accuracy 0.503):

| Condition | Accuracy |
|---|---|
| Clean | 0.503 |
| Volume × 0.5 / × 1.5 | 0.500 / 0.487 |
| Clip shortened to 75 % | 0.422 |
| Clip stretched to 125 % | 0.377 |
| 0.25 s silence added | 0.346 |
| Loudness normalised | 0.394 |
| **Gaussian noise (0.02)** | **0.195** |

Noise was the biggest weakness, so the model was retrained on clean + noisy copies (10,304 samples):

| Model | Clean test | Noisy test |
|---|---|---|
| Baseline | 0.503 | **0.195** |
| Noise-augmented | 0.469 | **0.443** |

Noise augmentation more than doubles accuracy on noisy audio, at a small cost on clean audio.

---

## LLM: persona and streaming

- `src/llm/ollama_client.py`: streams tokens from Llama via Ollama (`"stream": true`, `num_predict: 80`)
- `src/llm/emotion_prompt.py`: "Auralis" persona: concise, empathetic, 1–2 short sentences, no bullet points, never mentions the emotion model. The detected emotion is passed as **possibly imperfect** context.
- `src/pipeline.py`: audio file + transcript → emotion → streamed reply

The backend uses an async version of this client (same persona and rules) for the live system.

---

## Getting Started

```bash
git clone https://github.com/Ankit-builds1/Auralis.git
cd Auralis
git checkout ML
cd ml

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

**Trained models are not in git** (too large; `models/*.joblib` is gitignored). Place these in `ml/models/`:

| File | Size |
|---|---|
| `day7_temporal_svm_rbf.joblib` | ~94 MB |
| `day7_temporal_scaler.joblib` | ~57 KB |

**Dataset:** download [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) `AudioWAV` into `ml/data/raw/CREMA-D/AudioWAV/`.

Run (from the `ml/` folder):

```bash
# Predict emotion for one audio file
python -m src.models.audio_emotion

# Full pipeline: audio + transcript → emotion → Llama reply (needs Ollama running)
ollama pull llama3.2:3b
python -m src.pipeline
```

Retrain the deployed model:

```bash
python -m src.data.create_metadata
python -m src.data.split_dataset
python -m src.features.extract_temporal_features
python -m src.models.train_day7_temporal
```

---

## Project Structure

```
ml/
├── data/
│   ├── raw/CREMA-D/AudioWAV/       # dataset (download separately)
│   ├── metadata/                   # labels.csv + actor-independent splits
│   └── features/                   # generated features (gitignored)
├── models/                         # trained .joblib files (gitignored)
├── notebooks/                      # audio exploration
├── results/                        # metrics, confusion matrices, error analysis
├── src/
│   ├── data/                       # metadata + splitting
│   ├── features/                   # MFCC, spectral, prosodic, temporal features
│   ├── augmentation/               # noise augmentation + evaluation
│   ├── robustness/                 # robustness tests
│   ├── models/                     # training, tuning, inference, error analysis
│   ├── llm/                        # persona prompt + streaming Ollama client
│   └── pipeline.py                 # audio + transcript → emotion → reply
└── requirements.txt
```

---

## Progress

**Week 1 ✅** CREMA-D exploration, metadata, actor-independent splits, MFCC baseline

**Week 2 ✅** Advanced, prosodic and temporal features; SVM tuning; class weights; robustness tests; noise-augmented training; temporal SVM; error analysis; streaming Llama client + emotion-aware prompt

**Week 3 🔄** Inference pipeline, confidence analysis, end-to-end ML pipeline; connecting the emotion model to the live backend; emotion-conditioned TTS

---

## Known Limitations

- ~49 % accuracy on 6 emotions; CREMA-D is **acted studio speech**, so real laptop-mic voices will be harder
- Disgust and fear are the weakest classes
- Very sensitive to noise and added silence unless the noise-augmented model is used
- The deployed temporal model (0.490) is slightly below the combined temporal + prosodic model (0.503); worth re-checking which to deploy
- Model files must be shared manually (not in git)
