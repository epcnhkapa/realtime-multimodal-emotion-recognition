"""
Central configuration for the multimodal emotion recognition app.
All paths are RELATIVE to the project root (avoids unicode issues with
Vosk on Windows when the absolute path contains non-ASCII chars).
Run the app from the project root.
"""

# ---------- Model paths ----------
SER_PATH = 'models/ser_cnn_bilstm.keras'
VISION_PATH = 'models/vision_ferplus.keras'
NLP_PATH = 'models/mixed_nlp.keras'
VOSK_PATHS = {
    'en': 'models/vosk/vosk-model-small-en-us-0.15',
    'tr': 'models/vosk/vosk-model-small-tr-0.3',
}

# ---------- Class labels ----------
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NLP_CLASSES = ['Negative', 'Neutral', 'Positive']

# ---------- Audio constants ----------
SR = 16000                  # sample rate for both VAD and SER
VAD_CHUNK = 512             # 32 ms at 16 kHz, silero VAD chunk size

# SER feature extraction (must match training)
SER_DURATION = 3.0
SER_TARGET_LEN = int(SR * SER_DURATION)
SER_N_MELS = 128
SER_N_FFT = 1024
SER_HOP = 256
SER_TRIM_TOP_DB = 30

# ---------- VAD state machine ----------
SPEECH_START_THRESH = 0.5
SPEECH_END_THRESH = 0.3
SILENCE_CHUNKS_TO_END = 8         # 256 ms
SILENCE_CHUNKS_AFTER_LONG = 2     # 64 ms aggressive
LONG_SPEECH_THRESHOLD = 47        # 1.5 s
MIN_SPEECH_CHUNKS = 20            # 640 ms (unchanged)
MAX_SPEECH_CHUNKS = 94            # 3.0 s hard cap (matches training)

# ---------- Vision constants ----------
VISION_IMG_SIZE = 48
VIDEO_PREDICT_INTERVAL = 0.25      # seconds between vision inferences

# ---------- Fusion params ----------
W_VISION = 0.40
W_SER = 0.45
W_NLP_PEAK = 0.15
NLP_HALF_LIFE = 2.0               # seconds
SER_HALF_LIFE = 2.0               # seconds
VISION_FORGET_AFTER = 5.0         # seconds with no face -> clear vision
NLP_CONF_THRESH = 0.60            # ignore NLP if top prob below this

# Vision EWMA smoothing factor: 1.0 = no smoothing (jittery),
# 0.3 = balanced (~3-5 frame memory), 0.15 = very smooth (~10 frame memory)
VISION_SMOOTHING_ALPHA = 0.3


# NLP 3-class -> 6-class emotion boost mapping
# Rows: NLP class index (Neg, Neu, Pos)
# Cols: emotion index (Angry, Fear, Happy, Neutral, Sad, Surprise)
import numpy as np
NLP_TO_EMOTION = np.array([
    [0.33, 0.33, 0.00, 0.00, 0.33, 0.00],  # Negative
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00],  # Neutral
    [0.00, 0.00, 0.65, 0.00, 0.00, 0.35],  # Positive
], dtype=np.float32)

# ---------- UI ----------
FUSION_UPDATE_INTERVAL_MS = 200   # how often the GUI fusion display refreshes
WEBCAM_DEVICE_INDEX = 0
DEFAULT_LANG = 'en'
