"""
Simple SER test: press Enter -> record 3s -> predict emotion.
Ctrl+C to quit.
"""
import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'ser_cnn_bilstm.keras')
CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

SR = 16000
DURATION = 3.0
TARGET_LEN = int(SR * DURATION)
N_MELS = 128
N_FFT = 1024
HOP = 256

print('Loading model...')
model = tf.keras.models.load_model(MODEL_PATH)
print(f'Model loaded. Input shape: {model.input_shape}')

# List available input devices
print('\nAvailable input devices:')
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        marker = ' (default)' if i == sd.default.device[0] else ''
        print(f'  [{i}] {dev["name"]}{marker}')

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

def record_and_predict():
    print(f'\nRecording {DURATION}s...')
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    y = audio.squeeze()
    print(f'Recorded {len(y)} samples, RMS={np.sqrt(np.mean(y**2)):.4f}')

    # Trim silence (same as training pipeline)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    print(f'After trim: {len(y_trimmed)} samples ({len(y_trimmed)/SR:.2f}s)')

    y_fixed = fix_length(y_trimmed)
    mel = waveform_to_logmel(y_fixed)[..., np.newaxis]

    probs = model.predict(mel[np.newaxis, ...], verbose=0)[0]

    print('\nPredictions:')
    for cls, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
        bar = '#' * int(p * 40)
        print(f'  {cls:10s} {p:.3f} {bar}')

print('\nPress Enter to start a recording. Ctrl+C to quit.')
try:
    while True:
        input('\n[Enter to record]')
        record_and_predict()
except KeyboardInterrupt:
    print('\nBye.')
