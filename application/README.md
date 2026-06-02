# Desktop Application

PySide6 desktop application that runs the multimodal emotion recognition
pipeline in real time.

## Prerequisites

- **Windows 11** (the app was developed and tested here; Linux/macOS likely
  work for the live tab but not all video-tab codecs are guaranteed)
- **Python 3.13** (other 3.11+ versions should also work but the project
  was developed and tested on 3.13)
- A working webcam and microphone for the Live tab
- About 4 GB of free disk space for dependencies and models

## Installation

From a terminal at the application root:

```
pip install -r requirements.txt
```

This installs TensorFlow, PyTorch (used by Silero VAD), PySide6, OpenCV,
MediaPipe, Vosk, librosa, sounddevice, and imageio-ffmpeg.

Model files are included under `models/`:

- `vision_ferplus.keras` — facial expression model
- `ser_cnn_bilstm.keras` — speech emotion model
- `mixed_nlp.keras` — sentiment model
- `blaze_face_full_range.tflite` — MediaPipe face detector
- `vosk/vosk-model-small-en-us-0.15/` — Vosk English STT
- `vosk/vosk-model-small-tr-0.3/` — Vosk Turkish STT

## Running

From the project root:

```
run.bat
```

or directly:

```
py -3.13 -X utf8 main.py
```

The `-X utf8` flag is required because the NLP model has a Turkish
vocabulary and Windows' default cp1254 encoding breaks the load. The
`run.bat` script sets this up automatically.

## Usage

### Live tab

1. Press **Start**
2. Look at the webcam and start speaking
3. The top of the window shows the fused emotion
4. The right panel shows per-modality results (Vision continuously, SER
   and NLP after each utterance)
5. The bottom shows the live transcript and the last completed sentence
6. **Stop** ends capture cleanly

### Video File tab

1. Press **Load Video...** and pick an `.mp4`, `.avi`, `.mov`, `.mkv`,
   or `.webm`
2. Press **Play**
3. The video plays with synced audio (Qt media framework) while the
   pipeline processes audio and video frames in parallel
4. The seek slider can be dragged to skip around; the audio pipeline
   reseeks automatically
5. **Pause** / **Resume** / **Stop** behave as expected

The language selector (English / Türkçe) controls the Vosk model used for
speech-to-text. NLP works on both languages without needing to switch.

## Project structure

```
application/
├── main.py                  app entry point
├── run.bat                  Windows launcher with UTF-8 enforced
├── requirements.txt         Python dependencies
├── src/
│   ├── config.py            all constants (thresholds, paths, weights)
│   ├── models_loader.py     single point of model loading
│   ├── audio_utils.py       mel spectrogram, fix-length, PCM conversion
│   ├── pipeline.py          per-tab state container + fusion logic
│   ├── face_detection.py    MediaPipe face detection wrapper
│   ├── video_thread.py      live webcam thread
│   ├── audio_thread.py      live mic + VAD + STT + SER + NLP thread
│   ├── video_file_thread.py audio analysis thread for video file tab
│   └── ui/
│       ├── widgets.py       reusable UI components
│       ├── live_tab.py      Tab 1: live demo
│       ├── video_tab.py     Tab 2: video file demo
│       └── main_window.py   main window with tabs
├── tests/                   standalone scripts used during development
│   ├── vision_live_test.py
│   ├── ser_live_test.py
│   ├── ser_dataset_test.py
│   ├── vad_ser_test.py
│   ├── vad_ser_stt_test.py
│   ├── nlp_test.py
│   └── fusion_test.py
└── models/
    ├── vision_ferplus.keras
    ├── ser_cnn_bilstm.keras
    ├── mixed_nlp.keras
    ├── blaze_face_full_range.tflite
    └── vosk/
        ├── vosk-model-small-en-us-0.15/
        └── vosk-model-small-tr-0.3/
```

## Tuning

Frequently tuned values are all in `src/config.py`:

- **Fusion weights** (`W_VISION`, `W_SER`, `W_NLP_PEAK`)
- **Decay half-lives** (`SER_HALF_LIFE`, `NLP_HALF_LIFE`)
- **VAD thresholds** (`SPEECH_START_THRESH`, `SPEECH_END_THRESH`,
  `SILENCE_CHUNKS_TO_END`, `LONG_SPEECH_THRESHOLD`, `MAX_SPEECH_CHUNKS`)
- **Vision update rate** (`VIDEO_PREDICT_INTERVAL`)
- **NLP confidence floor** (`NLP_CONF_THRESH`)

Adjust, save, restart the app.

## Known limitations

- TensorFlow GPU support is not available on native Windows for
  TensorFlow >= 2.11. The app runs on CPU. Vision and NLP inference are
  fast enough; SER is the heaviest step but still under 100 ms.
- The Vosk small models are real-time on CPU but lower-quality than the
  large models. They are good enough for short utterances; long, fast
  speech may be transcribed imperfectly.
- The face crop used by the Vision model is slightly tightened from the
  MediaPipe bounding box to better match FER+ training distribution.
- The SER model expects exactly 3 seconds of audio at inference; longer
  utterances are center-cropped to 3 seconds. The VAD hard cap is set to
  3 seconds to match this and avoid discarding tail audio.
