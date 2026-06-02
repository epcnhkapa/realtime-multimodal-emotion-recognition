"""
VAD + SER + Vosk STT integration.
Language: 'en' (default) or 'tr' via CLI arg.
  py -3.13 tests/vad_ser_stt_test.py tr

IMPORTANT: Run from project root, not from tests/ folder:
  cd C:\\path\\to\\final_project_v2
  py -3.13 tests/vad_ser_stt_test.py
"""
import os
import sys
import time
import json
import collections

# Vosk first, then TF/Torch, to avoid native lib conflicts on Windows
from vosk import Model as VoskModel, KaldiRecognizer

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import torch
from silero_vad import load_silero_vad


# Relative paths only — avoid absolute paths because Vosk/Kaldi on Windows
# fails on non-ASCII path components (e.g. "Masaüstü").
SER_PATH = 'models/ser_cnn_bilstm.keras'
VOSK_PATHS = {
    'en': 'models/vosk/vosk-model-small-en-us-0.15',
    'tr': 'models/vosk/vosk-model-small-tr-0.3',
}

lang = sys.argv[1] if len(sys.argv) > 1 else 'en'
if lang not in VOSK_PATHS:
    raise SystemExit(f'Unknown lang {lang}. Use en or tr.')
vosk_path = VOSK_PATHS[lang]

# Sanity check: script must be run from project root
if not os.path.isdir(vosk_path):
    raise SystemExit(
        f'Vosk model not found at "{vosk_path}".\n'
        f'Make sure you run this script from the PROJECT ROOT:\n'
        f'  cd <project_root>\n'
        f'  py -3.13 tests/vad_ser_stt_test.py\n'
        f'Current working dir: {os.getcwd()}'
    )
if not os.path.isfile(SER_PATH):
    raise SystemExit(f'SER model not found at "{SER_PATH}". cwd={os.getcwd()}')


CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
SR = 16000
VAD_CHUNK = 512

SPEECH_START_THRESH = 0.5
SPEECH_END_THRESH = 0.3
SILENCE_CHUNKS_TO_END = 12
SILENCE_CHUNKS_AFTER_LONG = 6
LONG_SPEECH_THRESHOLD = 156
MIN_SPEECH_CHUNKS = 20
MAX_SPEECH_CHUNKS = 219

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

def float_to_pcm16_bytes(chunk_f32):
    clipped = np.clip(chunk_f32, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


print(f'Loading Vosk model [{lang}]...')
vosk_model = VoskModel(vosk_path)
rec = KaldiRecognizer(vosk_model, SR)
rec.SetWords(True)

print('Loading SER model...')
ser_model = tf.keras.models.load_model(SER_PATH)

print('Loading silero VAD...')
vad_model = load_silero_vad()

print('All loaded.\n')


q = collections.deque()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f'[audio status] {status}', file=sys.stderr)
    q.append(indata.copy().squeeze())


def chunk_iter():
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


def _clear_line():
    sys.stdout.write('\r' + ' ' * 100 + '\r')
    sys.stdout.flush()


def print_partial(text):
    _clear_line()
    sys.stdout.write(f'  ~ {text}')
    sys.stdout.flush()


def main():
    print(f'Listening at {SR} Hz, lang={lang}. Speak. Ctrl+C to quit.\n')

    state = 'SILENT'
    speech_buffer = []
    silent_run = 0
    last_partial = ''

    with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                        blocksize=VAD_CHUNK, callback=audio_callback):
        try:
            for chunk in chunk_iter():
                pcm_bytes = float_to_pcm16_bytes(chunk)
                rec.AcceptWaveform(pcm_bytes)

                with torch.no_grad():
                    prob = vad_model(torch.from_numpy(chunk), SR).item()

                if state == 'SILENT':
                    if prob >= SPEECH_START_THRESH:
                        state = 'SPEECH'
                        speech_buffer = [chunk]
                        silent_run = 0
                        last_partial = ''
                        print('[speech start]')
                else:
                    speech_buffer.append(chunk)

                    partial = json.loads(rec.PartialResult()).get('partial', '').strip()
                    if partial and partial != last_partial:
                        print_partial(partial)
                        last_partial = partial

                    if prob < SPEECH_END_THRESH:
                        silent_run += 1
                    else:
                        silent_run = 0

                    utterance_len = len(speech_buffer)
                    silence_threshold = (SILENCE_CHUNKS_AFTER_LONG
                                         if utterance_len >= LONG_SPEECH_THRESHOLD
                                         else SILENCE_CHUNKS_TO_END)

                    if silent_run >= silence_threshold or utterance_len >= MAX_SPEECH_CHUNKS:
                        audio = np.concatenate(speech_buffer).astype(np.float32)
                        duration = len(audio) / SR
                        state = 'SILENT'
                        speech_buffer = []
                        silent_run = 0

                        final_text = json.loads(rec.FinalResult()).get('text', '').strip()

                        _clear_line()
                        if utterance_len < MIN_SPEECH_CHUNKS:
                            print(f'[speech end] too short ({duration:.2f}s), ignored\n')
                            continue

                        print(f'[speech end] duration {duration:.2f}s')
                        print(f'  transcript: "{final_text}"')

                        probs = predict_emotion(audio)
                        if probs is None:
                            print('  (SER: too short after trim)\n')
                            continue
                        top = int(np.argmax(probs))
                        print(f'  emotion:    {CLASSES[top]} ({probs[top]:.3f})')
                        top3 = sorted(zip(CLASSES, probs), key=lambda x: -x[1])[:3]
                        top3_str = ', '.join(f'{c} {p:.2f}' for c, p in top3)
                        print(f'  top3:       {top3_str}\n')
        except KeyboardInterrupt:
            print('\nBye.')


if __name__ == '__main__':
    main()