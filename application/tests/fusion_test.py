"""
Full multimodal fusion integration test.

Threads:
  Video thread:  webcam -> Vision model -> state.vision_probs
  Main thread:   mic -> VAD + STT in parallel, on speech-end -> SER + NLP -> state

Fusion every ~200ms:
  final = w_v * vision + w_a * ser + w_n * nlp_mapped
  w_n decays exponentially since last NLP result (half-life 2.5s)
  NLP ignored if its top confidence < 0.60

Language: en (default) or tr
  py -3.13 -X utf8 tests/fusion_test.py tr

Run from project root.
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

import sys
import time
import json
import threading
import collections

from vosk import Model as VoskModel, KaldiRecognizer

import numpy as np
import cv2
import sounddevice as sd
import librosa
import tensorflow as tf
import torch
from silero_vad import load_silero_vad


# ---------- Paths (relative to avoid Vosk unicode bug) ----------
SER_PATH = 'models/ser_cnn_bilstm.keras'
VISION_PATH = 'models/vision_ferplus.keras'
NLP_PATH = 'models/mixed_nlp.keras'
VOSK_PATHS = {
    'en': 'models/vosk/vosk-model-small-en-us-0.15',
    'tr': 'models/vosk/vosk-model-small-tr-0.3',
}

lang = sys.argv[1] if len(sys.argv) > 1 else 'en'
if lang not in VOSK_PATHS:
    raise SystemExit(f'Unknown lang {lang}. Use en or tr.')

for p in [SER_PATH, VISION_PATH, NLP_PATH]:
    if not os.path.isfile(p):
        raise SystemExit(f'Missing model file: {p} (cwd={os.getcwd()})')
if not os.path.isdir(VOSK_PATHS[lang]):
    raise SystemExit(f'Missing vosk model: {VOSK_PATHS[lang]}')


# ---------- Constants ----------
CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NLP_CLASSES = ['Negative', 'Neutral', 'Positive']

SR = 16000
VAD_CHUNK = 512
SPEECH_START_THRESH = 0.5
SPEECH_END_THRESH = 0.3
SILENCE_CHUNKS_TO_END = 12
SILENCE_CHUNKS_AFTER_LONG = 6
LONG_SPEECH_THRESHOLD = 156
MIN_SPEECH_CHUNKS = 20
MAX_SPEECH_CHUNKS = 219

SER_TARGET_LEN = int(SR * 3.0)
N_MELS, N_FFT, HOP = 128, 1024, 256

IMG_SIZE = 48
VIDEO_PREDICT_INTERVAL = 0.1   # s between vision inferences

# Fusion params
W_VISION = 0.50
W_SER = 0.35
W_NLP_PEAK = 0.20
NLP_HALF_LIFE = 2.5       # s
NLP_CONF_THRESH = 0.60
FUSION_PRINT_INTERVAL = 0.5  # s between terminal prints

# NLP 3-class -> 6-class emotion boost
# Row order: Negative, Neutral, Positive
# Col order: Angry, Fear, Happy, Neutral, Sad, Surprise
NLP_TO_EMOTION = np.array([
    [0.33, 0.33, 0.00, 0.00, 0.33, 0.00],  # Negative
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00],  # Neutral
    [0.00, 0.00, 0.70, 0.00, 0.00, 0.30],  # Positive
], dtype=np.float32)


# ---------- Utility ----------
def fix_length(y, target_len=SER_TARGET_LEN):
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

def float_to_pcm16_bytes(chunk_f32):
    clipped = np.clip(chunk_f32, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()

def nlp_to_6class(nlp_probs):
    return nlp_probs @ NLP_TO_EMOTION  # (3,) @ (3,6) -> (6,)


# ---------- Load models ----------
print(f'Loading Vosk [{lang}]...')
vosk_model = VoskModel(VOSK_PATHS[lang])
rec = KaldiRecognizer(vosk_model, SR)
rec.SetWords(True)

print('Loading Vision...')
vision_model = tf.keras.models.load_model(VISION_PATH)

print('Loading SER...')
ser_model = tf.keras.models.load_model(SER_PATH)

print('Loading NLP...')
nlp_model = tf.keras.models.load_model(NLP_PATH)

print('Loading silero VAD...')
vad_model = load_silero_vad()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Warm up models so first real inference is not slow
_ = vision_model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32), verbose=0)
_ = ser_model.predict(np.zeros((1, N_MELS, 188, 1), dtype=np.float32), verbose=0)
_ = nlp_model(tf.constant(['warmup']), training=False)

print('All loaded.\n')


# ---------- Shared state ----------
state_lock = threading.Lock()
state = {
    'vision_probs': None,
    'ser_probs': None,
    'nlp_probs': None,
    'nlp_timestamp': 0.0,
    'last_transcript': '',
    'stop': False,
}


# ---------- Video thread ----------
def video_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[video] cannot open webcam', file=sys.stderr)
        with state_lock:
            state['stop'] = True
        return
    last_predict = 0.0
    while not state['stop']:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        now = time.time()
        if now - last_predict >= VIDEO_PREDICT_INTERVAL:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi = roi.astype(np.float32) / 255.0
                roi = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                probs = vision_model.predict(roi, verbose=0)[0]
                with state_lock:
                    state['vision_probs'] = probs
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                top = int(np.argmax(probs))
                cv2.putText(frame, f'{CLASSES[top]} {probs[top]:.2f}',
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            last_predict = now
        cv2.imshow('Fusion test - press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            with state_lock:
                state['stop'] = True
            break
    cap.release()
    cv2.destroyAllWindows()


# ---------- Audio capture (callback) ----------
audio_q = collections.deque()
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f'[audio status] {status}', file=sys.stderr)
    audio_q.append(indata.copy().squeeze())


# ---------- Fusion ----------
def compute_fusion(now):
    with state_lock:
        v = state['vision_probs']
        a = state['ser_probs']
        n = state['nlp_probs']
        nlp_t = state['nlp_timestamp']

    if v is None:
        return None

    w_v = W_VISION
    w_a = W_SER if a is not None else 0.0

    w_n = 0.0
    if n is not None:
        n_conf = float(n.max())
        if n_conf >= NLP_CONF_THRESH:
            dt = now - nlp_t
            w_n = W_NLP_PEAK * (0.5 ** (dt / NLP_HALF_LIFE))

    total = w_v + w_a + w_n
    w_v, w_a, w_n = w_v/total, w_a/total, w_n/total

    result = w_v * v
    if a is not None:
        result = result + w_a * a
    if n is not None and w_n > 0:
        result = result + w_n * nlp_to_6class(n)

    s = result.sum()
    if s > 0:
        result = result / s
    return result, (w_v, w_a, w_n)


# ---------- Main audio + pipeline loop ----------
def audio_main_loop():
    state_machine = 'SILENT'
    speech_buffer = []
    silent_run = 0
    last_partial = ''
    last_fusion_print = 0.0

    with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                        blocksize=VAD_CHUNK, callback=audio_callback):
        buf = np.empty(0, dtype=np.float32)
        while not state['stop']:
            # Pull enough samples for one VAD chunk
            while len(buf) < VAD_CHUNK and not state['stop']:
                if audio_q:
                    buf = np.concatenate([buf, audio_q.popleft()])
                else:
                    time.sleep(0.005)
            if state['stop']:
                break
            chunk = buf[:VAD_CHUNK]
            buf = buf[VAD_CHUNK:]

            # Feed Vosk
            rec.AcceptWaveform(float_to_pcm16_bytes(chunk))

            # VAD
            with torch.no_grad():
                prob = vad_model(torch.from_numpy(chunk), SR).item()

            if state_machine == 'SILENT':
                if prob >= SPEECH_START_THRESH:
                    state_machine = 'SPEECH'
                    speech_buffer = [chunk]
                    silent_run = 0
                    last_partial = ''
                    print('\n[speech start]')
            else:
                speech_buffer.append(chunk)

                partial = json.loads(rec.PartialResult()).get('partial', '').strip()
                if partial and partial != last_partial:
                    sys.stdout.write(f'\r  ~ {partial}' + ' ' * 20)
                    sys.stdout.flush()
                    last_partial = partial

                if prob < SPEECH_END_THRESH:
                    silent_run += 1
                else:
                    silent_run = 0

                ulen = len(speech_buffer)
                silence_threshold = (SILENCE_CHUNKS_AFTER_LONG
                                     if ulen >= LONG_SPEECH_THRESHOLD
                                     else SILENCE_CHUNKS_TO_END)

                if silent_run >= silence_threshold or ulen >= MAX_SPEECH_CHUNKS:
                    audio = np.concatenate(speech_buffer).astype(np.float32)
                    duration = len(audio) / SR
                    state_machine = 'SILENT'
                    speech_buffer = []
                    silent_run = 0
                    final_text = json.loads(rec.FinalResult()).get('text', '').strip()
                    sys.stdout.write('\r' + ' ' * 80 + '\r')

                    if ulen < MIN_SPEECH_CHUNKS:
                        print(f'[speech end] too short ({duration:.2f}s), ignored')
                        continue

                    print(f'[speech end] {duration:.2f}s  transcript: "{final_text}"')

                    # SER
                    audio_trim, _ = librosa.effects.trim(audio, top_db=30)
                    if len(audio_trim) >= SR * 0.3:
                        audio_fixed = fix_length(audio_trim)
                        mel = waveform_to_logmel(audio_fixed)[..., np.newaxis]
                        ser_probs = ser_model.predict(mel[np.newaxis, ...], verbose=0)[0]
                        with state_lock:
                            state['ser_probs'] = ser_probs
                        top = int(np.argmax(ser_probs))
                        print(f'  SER: {CLASSES[top]} ({ser_probs[top]:.2f})')

                    # NLP
                    if final_text:
                        nlp_probs = nlp_model(tf.constant([final_text]), training=False).numpy()[0]
                        with state_lock:
                            state['nlp_probs'] = nlp_probs
                            state['nlp_timestamp'] = time.time()
                            state['last_transcript'] = final_text
                        top = int(np.argmax(nlp_probs))
                        print(f'  NLP: {NLP_CLASSES[top]} ({nlp_probs[top]:.2f})')

            # Fusion print every 500 ms
            now = time.time()
            if now - last_fusion_print >= FUSION_PRINT_INTERVAL:
                out = compute_fusion(now)
                if out is not None:
                    fused, (wv, wa, wn) = out
                    top = int(np.argmax(fused))
                    top3 = sorted(zip(CLASSES, fused), key=lambda x: -x[1])[:3]
                    top3_str = ', '.join(f'{c} {p:.2f}' for c, p in top3)
                    print(f'[fusion] {CLASSES[top]:10s} ({fused[top]:.2f})  '
                          f'w=(v{wv:.2f} a{wa:.2f} n{wn:.2f})  top3: {top3_str}')
                last_fusion_print = now


def main():
    print('Starting video thread...')
    vt = threading.Thread(target=video_loop, daemon=True)
    vt.start()
    time.sleep(0.5)  # let the webcam warm up

    print('Starting audio/fusion loop. Look at the camera, speak, press Q in video window to quit.\n')
    try:
        audio_main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        with state_lock:
            state['stop'] = True
        vt.join(timeout=1.0)
        print('\nBye.')


if __name__ == '__main__':
    main()
