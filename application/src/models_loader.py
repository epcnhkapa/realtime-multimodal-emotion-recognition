"""
Single point of model loading. Loads each model once at app startup
and exposes them as module-level attributes.

Order matters on Windows: Vosk loads first to avoid native lib conflicts
with TensorFlow / PyTorch.
"""
import os
import numpy as np

from src import config


# Loaded lazily so importing this module doesn't trigger heavy work.
# Call load_all() once at app startup.
vosk_models = {}      # lang -> VoskModel
vision_model = None
ser_model = None
nlp_model = None
vad_model = None


def _check_paths():
    for p in [config.SER_PATH, config.VISION_PATH, config.NLP_PATH]:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f'Missing model file: {p}\n'
                f'Run from project root. cwd={os.getcwd()}')
    for lang, path in config.VOSK_PATHS.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(f'Missing Vosk [{lang}] model dir: {path}')


def load_all(verbose=True):
    """Load every model. Call once before any inference."""
    global vision_model, ser_model, nlp_model, vad_model

    _check_paths()

    # Vosk first
    from vosk import Model as VoskModel, SetLogLevel
    SetLogLevel(-1)
    for lang, path in config.VOSK_PATHS.items():
        if verbose:
            print(f'Loading Vosk [{lang}]...')
        vosk_models[lang] = VoskModel(path)

    # TF models
    import tensorflow as tf
    if verbose:
        print('Loading Vision...')
    vision_model = tf.keras.models.load_model(config.VISION_PATH)

    if verbose:
        print('Loading SER...')
    ser_model = tf.keras.models.load_model(config.SER_PATH)

    if verbose:
        print('Loading NLP...')
    nlp_model = tf.keras.models.load_model(config.NLP_PATH)

    # Silero VAD (PyTorch under the hood)
    if verbose:
        print('Loading silero VAD...')
    from silero_vad import load_silero_vad
    vad_model = load_silero_vad()


    # Warm up TF models so first inference is fast
    if verbose:
        print('Warming up...')
    _ = vision_model.predict(
        np.zeros((1, config.VISION_IMG_SIZE, config.VISION_IMG_SIZE, 1),
                 dtype=np.float32),
        verbose=0)
    _ = ser_model.predict(
        np.zeros((1, config.SER_N_MELS, 188, 1), dtype=np.float32),
        verbose=0)
    _ = nlp_model(tf.constant(['warmup']), training=False)

    if verbose:
        print('All models loaded.')


def make_recognizer(lang):
    """Create a fresh KaldiRecognizer for the given language.
    Each pipeline (Tab1 / Tab2) gets its own recognizer to keep state
    independent, but they share the underlying VoskModel."""
    from vosk import KaldiRecognizer
    if lang not in vosk_models:
        raise ValueError(f'Unknown lang: {lang}')
    rec = KaldiRecognizer(vosk_models[lang], config.SR)
    rec.SetWords(True)
    return rec
