# Realtime Multimodal Emotion Recognition

A real-time emotion recognition system that fuses three modalities — facial
expression, speech, and text — into a single emotion estimate. Built as a
desktop application with a PySide6 GUI on Windows 11, Python 3.13, and
TensorFlow 2.21.

This repository contains both the **end-user application** and the
**Colab notebooks** used to train the underlying models, so the full
pipeline (data → trained models → real-time inference) can be inspected
or reproduced.

## Demo at a glance

The system has two tabs in a single window:

- **Live**: streams from the webcam and microphone in real time.
- **Video File**: loads any local video and analyzes it the same way.

For each utterance the system shows:

- Top emotion (fused decision)
- Per-modality breakdown: Vision (face), SER (speech), NLP (text)
- Live word-by-word transcript and the finalized last sentence

## Repository layout

```
realtime-multimodal-emotion-recognition/
├── application/        Desktop app (PySide6 GUI)
│   ├── main.py
│   ├── run.bat
│   ├── requirements.txt
│   ├── benchmark.py            performance benchmark script
│   ├── benchmark_results.json  measured inference latencies
│   ├── src/                    modules: pipeline, audio, video, UI, ...
│   ├── tests/                  standalone test scripts for each model
│   └── models/                 trained models and Vosk STT data
└── colab_notebooks/    training notebooks for Vision, SER, NLP
```

## Models in the pipeline

| Modality | Architecture | Training data | Test accuracy |
|---|---|---|---|
| Vision (FER) | Custom CNN with augmentation, label smoothing, mixup | FER2013 + FER+ soft labels | ~79.4% |
| SER (audio) | CNN + BiLSTM on log-mel spectrograms | RAVDESS + CREMA-D (speaker-independent split) | ~69.4% |
| NLP (text) | TextVectorization + Dense | Mixed English/Turkish sentiment data | ~72% |

The Vision and SER models share six emotion classes:
**Angry, Fear, Happy, Neutral, Sad, Surprise.**
NLP outputs three classes (Negative, Neutral, Positive) which are mapped
to the six-class space inside the fusion module.

## Fusion approach

Modalities run on different time scales: Vision is continuous, SER and NLP
update at utterance boundaries (driven by Silero VAD). The fusion layer is
a weighted average:

- Vision: continuous, weight 0.50
- SER: per-utterance, weight 0.35, exponential decay (half-life 2.5 s)
- NLP: per-utterance, weight 0.15, exponential decay (half-life 2 s),
  ignored entirely below 0.60 confidence
- Vision is also forgotten if no face has been seen for 5 s

NLP's 3-class output is mapped to the 6-class emotion space through a
fixed boost matrix.

## Getting started

See [`application/README.md`](application/README.md) for setup,
dependencies, and how to run the desktop app.
See [`colab_notebooks/README.md`](colab_notebooks/README.md) for the
training notebooks.

## License

MIT
