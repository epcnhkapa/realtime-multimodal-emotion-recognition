"""
VAD test: listen continuously, detect speech segments, predict emotion on each.

Silero VAD reads 16kHz mono audio in 512-sample chunks (32 ms each).
State machine: SILENT -> (VAD high) -> SPEECH -> (VAD low for N frames) -> SILENT

Press Ctrl+C to quit.
"""
import os
import sys
import time
import collections
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import torch
from silero_vad import load_silero_vad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'models', 'ser_cnn_bilstm.keras')

CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
SR = 16000
VAD_CHUNK = 512  # 32 ms at 16 kHz

# Hysteresis thresholds
SPEECH_START_THRESH = 0.5
SPEECH_END_THRESH = 0.3
SILENCE_CHUNKS_TO_END = 12      # 400 ms — normal bitiş
SILENCE_CHUNKS_AFTER_LONG = 6   # 200 ms — uzun konuşmada daha agresif
LONG_SPEECH_THRESHOLD = 156     # 5.0 s — bu süreden sonra agresif bitişe geç
MIN_SPEECH_CHUNKS = 20          # 640 ms
MAX_SPEECH_CHUNKS = 219         # 7.0 s — hard cap
# SER preprocessing params (must match training)
TARGET_LEN = int(SR * 3.0)
N_MELS = 128
N_FFT = 1024
HOP = 256


def fix_length(y, target_len=TARGET_LEN):
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]
    elif len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y

def waveform_to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)


print('Loading models...')
ser_model = tf.keras.models.load_model(MODEL_PATH)
vad_model = load_silero_vad()
print('Models loaded.')

# Audio queue — thread-safe deque of numpy arrays
q = collections.deque()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f'[audio status] {status}', file=sys.stderr)
    # indata is float32 (frames, 1); store flat copies
    q.append(indata.copy().squeeze())


def chunk_iter():
    """Yield fixed-size VAD_CHUNK arrays from the streaming input queue."""
    buf = np.empty(0, dtype=np.float32)
    while True:
        while len(buf) < VAD_CHUNK:
            if q:
                buf = np.concatenate([buf, q.popleft()])
            else:
                time.sleep(0.005)
        chunk = buf[:VAD_CHUNK]
        buf = buf[VAD_CHUNK:]
        yield chunk


def predict_emotion(audio):
    audio_trim, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio_trim) < SR * 0.3:
        return None
    audio_fixed = fix_length(audio_trim)
    mel = waveform_to_logmel(audio_fixed)[..., np.newaxis]
    probs = ser_model.predict(mel[np.newaxis, ...], verbose=0)[0]
    return probs


def main():
    print(f'\nListening on default mic at {SR} Hz. Speak a sentence. Ctrl+C to quit.\n')

    state = 'SILENT'
    speech_buffer = []      # list of np arrays (VAD_CHUNK size each)
    silent_run = 0          # consecutive silent chunks while in SPEECH state

    with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                        blocksize=VAD_CHUNK, callback=audio_callback):
        try:
            for chunk in chunk_iter():
                t = torch.from_numpy(chunk)
                # silero_vad expects float32 1D tensor, sr as int
                with torch.no_grad():
                    prob = vad_model(t, SR).item()

                if state == 'SILENT':
                    if prob >= SPEECH_START_THRESH:
                        state = 'SPEECH'
                        speech_buffer = [chunk]
                        silent_run = 0
                        print('[speech start]', flush=True)
                else:  # SPEECH
                    speech_buffer.append(chunk)
                    if prob < SPEECH_END_THRESH:
                        silent_run += 1
                    else:
                        silent_run = 0

                    utterance_len = len(speech_buffer)
                    # Decide silence threshold based on how long we've been speaking
                    if utterance_len >= LONG_SPEECH_THRESHOLD:
                        silence_threshold = SILENCE_CHUNKS_AFTER_LONG
                    else:
                        silence_threshold = SILENCE_CHUNKS_TO_END
                    if silent_run >= silence_threshold or utterance_len >= MAX_SPEECH_CHUNKS:
                        # End of utterance
                        audio = np.concatenate(speech_buffer).astype(np.float32)
                        duration = len(audio) / SR
                        state = 'SILENT'
                        speech_buffer = []
                        silent_run = 0

                        if utterance_len < MIN_SPEECH_CHUNKS:
                            print(f'[speech end] too short ({duration:.2f}s), ignored', flush=True)
                            continue

                        print(f'[speech end] duration {duration:.2f}s, running SER...', flush=True)
                        probs = predict_emotion(audio)
                        if probs is None:
                            print('  (after trim, too short to predict)', flush=True)
                            continue
                        top = int(np.argmax(probs))
                        print(f'  -> {CLASSES[top]} ({probs[top]:.3f})', flush=True)
                        for cls, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
                            bar = '#' * int(p * 30)
                            print(f'     {cls:10s} {p:.3f} {bar}')
                        print()
        except KeyboardInterrupt:
            print('\nBye.')


if __name__ == '__main__':
    main()
