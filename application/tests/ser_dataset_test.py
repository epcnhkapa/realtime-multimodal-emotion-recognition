"""
SER test on dataset samples (held-out speakers).
Reads all .wav files in assets/test_samples/ and predicts emotion.
Filename format: <Emotion>_<index>.wav  e.g. Angry_1.wav
"""
import os
import glob
import numpy as np
import librosa
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'models', 'ser_cnn_bilstm.keras')
SAMPLES_DIR = os.path.join(ROOT, 'assets', 'test_samples')

CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
SR = 16000
DURATION = 3.0
TARGET_LEN = int(SR * DURATION)
N_MELS = 128
N_FFT = 1024
HOP = 256

print('Loading model...')
model = tf.keras.models.load_model(MODEL_PATH)
print(f'Model loaded.\n')

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

def predict_file(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    y_fixed = fix_length(y_trim)
    mel = waveform_to_logmel(y_fixed)[..., np.newaxis]
    probs = model.predict(mel[np.newaxis, ...], verbose=0)[0]
    return probs, len(y), len(y_trim)

wav_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, '*.wav')))
if not wav_files:
    raise SystemExit(f'No .wav files in {SAMPLES_DIR}')

print(f'Found {len(wav_files)} files.\n')
print(f'{"File":30s} {"Expected":10s} {"Predicted":10s} {"Conf":6s}  Result')
print('-' * 80)

correct = 0
results = []
for path in wav_files:
    fname = os.path.basename(path)
    expected = fname.split('_')[0]  # e.g. Angry from Angry_1.wav
    if expected not in CLASSES:
        print(f'{fname:30s} (skipping, unknown class)')
        continue

    probs, orig_len, trim_len = predict_file(path)
    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    conf = float(probs[pred_idx])

    is_correct = pred_label == expected
    if is_correct:
        correct += 1
    mark = 'OK' if is_correct else 'WRONG'
    print(f'{fname:30s} {expected:10s} {pred_label:10s} {conf:.3f}  {mark}')

    results.append((fname, expected, pred_label, conf, probs))

total = len(results)
acc = correct / total if total else 0
print('-' * 80)
print(f'Accuracy: {correct}/{total} = {acc:.2%}\n')

# Detailed probs for the wrong ones
print('=' * 80)
print('Probability breakdown for WRONG predictions:')
print('=' * 80)
any_wrong = False
for fname, expected, pred, conf, probs in results:
    if pred == expected:
        continue
    any_wrong = True
    print(f'\n{fname}  expected={expected}  predicted={pred}')
    for cls, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
        bar = '#' * int(p * 30)
        marker = ''
        if cls == expected: marker = '  <- expected'
        if cls == pred:     marker = '  <- predicted'
        print(f'  {cls:10s} {p:.3f} {bar}{marker}')

if not any_wrong:
    print('\nAll predictions correct.')
